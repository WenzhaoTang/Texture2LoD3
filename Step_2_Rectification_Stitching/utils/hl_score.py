import numpy as np
from util.filter_verhor_lines import filter_verhor_lines
from vp_predict import vp_predict

def hl_score(hl_samp, ls_homo, z_homo, params):
    candidates = [{} for i in range(hl_samp.shape[1])]
    nhvps = []
    for i in range(hl_samp.shape[1]):
        helpfulIds = filter_verhor_lines(ls_homo, z_homo, params)
        initialIds = np.arange(len(helpfulIds))
        candidates[i]["horizon_homo"] = hl_samp[:, i]
        [candidates[i]["sc"], candidates[i]["hvp_homo"], hvp_groups] = vp_predict(ls_homo[:, helpfulIds], initialIds, candidates[i]["horizon_homo"], params)
        print("hvp_groups:", hvp_groups)
        print("helpfulIds length:", len(helpfulIds))

        # Filter valid indices based on the length of helpfulIds and store in candidates
        candidates[i]["hvp_groups"] = []
        for group in hvp_groups:
            # Check each index in group to ensure it is within the range of helpfulIds
            valid_indices = group[group < len(helpfulIds)]
            candidates[i]["hvp_groups"].extend([helpfulIds[k] for k in valid_indices])

        print("Filtered indices:", candidates[i]["hvp_groups"])
        nhvps.append(candidates[i]["hvp_homo"].shape[0])

    # Decide the horizon line
    horCandidateScores = np.array([candidates[i]["sc"] for i in range(hl_samp.shape[1])])
    maxHorCandidateId = np.argmax(horCandidateScores)
    hl_homo = candidates[maxHorCandidateId]["horizon_homo"]

    # Output results
    results = {}
    results["hvp_groups"] = candidates[maxHorCandidateId]["hvp_groups"]
    results["hvp_homo"] = candidates[maxHorCandidateId]["hvp_homo"]
    results["score"] = candidates[maxHorCandidateId]["sc"]

    return hl_homo, results





