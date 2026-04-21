

# Imports
import os
import glob
import mne
import pandas as pd
import statsmodels.formula.api as smf

from mne_nirs.experimental_design import make_first_level_design_matrix
from mne_nirs.statistics import run_glm

#---Paths---
hbRoot = "PATH TO PROCESSED HB FILES FROM optical_density_to_hb.py"
outDir = "PATH_TO_GLM_OUTPUT_FOLDER_HERE"
agePath = "TC_ALL_DOB.xlsx"
behavPath = "TC_Behavioral_Data.xlsx"

os.makedirs(outDir, exist_ok=True)

#---Glm Files---
hbFiles = sorted(
    glob.glob(os.path.join(hbRoot, "**", "*_hb_raw.fif"), recursive=True)
)

print("Found Hb files:", len(hbFiles))

if len(hbFiles) == 0:
    raise FileNotFoundError("No hb fif files found. Run previous pipeline step.")

#---Glm Run---
allResults = []

for hbPath in hbFiles:
    print(f"\nRunning GLM for: {hbPath}")

    raw = mne.io.read_raw_fif(hbPath, preload=True, verbose=False)

    designMatrix = make_first_level_design_matrix(
        raw,
        stim_dur=5.0,
        hrf_model="glover"
    )

    glmEst = run_glm(raw, designMatrix)
    glmDf = glmEst.to_dataframe()

    fileName = os.path.basename(hbPath)
    subjectId = fileName.split("_")[0]
    runId = fileName.split("_")[1]

    glmDf["subject"] = subjectId
    glmDf["run"] = runId
    glmDf["sourceFile"] = fileName

    subjectCsv = os.path.join(
        outDir,
        fileName.replace("_hb_raw.fif", "_glm_results.csv")
    )

    glmDf.to_csv(subjectCsv, index=False)
    allResults.append(glmDf)

#---Combine Results---
if len(allResults) == 0:
    raise ValueError("No GLM results created.")

groupDf = pd.concat(allResults, ignore_index=True)
groupCsv = os.path.join(outDir, "allSubjectsGlmResults.csv")
groupDf.to_csv(groupCsv, index=False)

#---Glm Table---
glmDf = groupDf[groupDf["Condition"].isin(["Condition_1", "Condition_2"])].copy()
glmDf = glmDf.rename(columns={"theta": "activation"})
glmDf["activationScaled"] = glmDf["activation"] * 1e6

#---Age Data---
ageDf = pd.read_excel(agePath)
ageDf = ageDf[["SubNum", "DOB", "DOS"]].copy()
ageDf["DOB"] = pd.to_datetime(ageDf["DOB"])
ageDf["DOS"] = pd.to_datetime(ageDf["DOS"])
ageDf["ageYears"] = (ageDf["DOS"] - ageDf["DOB"]).dt.days / 365.25
ageDf = ageDf.rename(columns={"SubNum": "subnum"})

glmDf["subnum"] = glmDf["subject"].str.extract(r"(\d+)$")[0]
glmDf["subnum"] = pd.to_numeric(glmDf["subnum"], errors="coerce")
glmDf["subnum"] = 1000 + glmDf["subnum"]

glmDf = glmDf.merge(
    ageDf[["subnum", "ageYears"]],
    on="subnum",
    how="left"
)

#---Behavior Data---
behavDf = pd.read_excel(behavPath)

behavUse = behavDf[["SubNum", "TType", "Acc"]].copy()
behavUse = behavUse.rename(columns={
    "SubNum": "subnum",
    "Acc": "accuracy"
})

behavUse["Condition"] = behavUse["TType"].map({
    1: "Condition_1",
    2: "Condition_2"
})

behavUse = (
    behavUse
    .dropna(subset=["Condition"])
    .groupby(["subnum", "Condition"], as_index=False)["accuracy"]
    .mean()
)

glmDf = glmDf.merge(
    behavUse[["subnum", "Condition", "accuracy"]],
    on=["subnum", "Condition"],
    how="left"
)

#---Final Table---
modelDf = glmDf.dropna(
    subset=["activationScaled", "ageYears", "Condition", "accuracy"]
).copy()

#---Hbo Model---
hboDf = modelDf[modelDf["Chroma"].str.lower() == "hbo"].copy()

hboModel = smf.mixedlm(
    "activationScaled ~ C(Condition) + ageYears + accuracy",
    data=hboDf,
    groups=hboDf["subject"]
)

hboResult = hboModel.fit(method="lbfgs")

print("\n===== HbO Model =====")
print(hboResult.summary())

#---Hbr Model---
hbrDf = modelDf[modelDf["Chroma"].str.lower() == "hbr"].copy()

hbrModel = smf.mixedlm(
    "activationScaled ~ C(Condition) + ageYears + accuracy",
    data=hbrDf,
    groups=hbrDf["subject"]
)

hbrResult = hbrModel.fit(method="lbfgs")

print("\n===== HbR Model =====")
print(hbrResult.summary())

#---Save Outputs---
modelOut = os.path.join(outDir, "finalGroupModelTable.csv")
hboOut = os.path.join(outDir, "hboModelSummary.txt")
hbrOut = os.path.join(outDir, "hbrModelSummary.txt")

modelDf.to_csv(modelOut, index=False)

with open(hboOut, "w") as file:
    file.write(hboResult.summary().as_text())

with open(hbrOut, "w") as file:
    file.write(hbrResult.summary().as_text())

print("\nAll outputs saved.")

# %%
#---Coverage Check---
print("Unique subjects:", modelDf["subject"].nunique())
print("Unique runs:", modelDf["run"].nunique())
print("Total rows:", len(modelDf))

# %%
#---Condition Model---
hboModel = smf.mixedlm(
    "activationScaled ~ C(Condition)",
    data=hboDf,
    groups=hboDf["subject"]
)

print(hboModel.fit().summary())

# %%
#---Age Model---
hboModel = smf.mixedlm(
    "activationScaled ~ ageYears",
    data=hboDf,
    groups=hboDf["subject"]
)

print(hboModel.fit().summary())

# %%
#---Accuracy Model---
hboModel = smf.mixedlm(
    "activationScaled ~ accuracy",
    data=hboDf,
    groups=hboDf["subject"]
)

print(hboModel.fit().summary())

# %%
