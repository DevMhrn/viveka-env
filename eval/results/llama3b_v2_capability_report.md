# Capability Report — llama3b_v2

Base: 20 scenarios | Trained: 20 scenarios (t12 + t34 combined). Source: inference JSONs in `eval/results/`.

## 1. Per-component reward (base → trained)

| Component | Weight | Base | Trained | Δ |
|---|---|---|---|---|
| reversibility_correct | 0.30 | 0.5282 | 0.4826 | **-0.0456** |
| task_progress | 0.25 | 0.6000 | 0.5750 | **-0.0250** |
| confirmation_appropriate | 0.15 | 0.6000 | 0.6000 | **+0.0000** |
| confidence_brier | 0.15 | 0.5189 | 0.4897 | **-0.0292** |
| over_asking | 0.10 | 0.9000 | 0.9750 | **+0.0750** |
| hallucination | 0.05 | 0.8300 | 0.8385 | **+0.0085** |

> `reversibility_correct` is the headline skill (weight 0.30). A positive Δ here is the most direct evidence the model learned to predict reversibility.

## 2. Per-tier reward + reversibility_correct

| Tier | n | Base reward | Trained reward | Δ reward | Base rev_correct | Trained rev_correct | Δ rev_correct |
|---|---|---|---|---|---|---|---|
| T1 | 5 | 0.3926 | 0.3331 | **-0.0594** | 0.6000 | 0.4000 | **-0.2000** |
| T2 | 5 | 0.1938 | 0.2417 | **+0.0479** | 0.3357 | 0.3926 | **+0.0569** |
| T3 | 5 | 0.2105 | 0.2096 | **-0.0009** | 0.4000 | 0.4000 | **+0.0000** |
| T4 | 5 | 0.0892 | 0.0882 | **-0.0010** | 0.7770 | 0.7378 | **-0.0392** |

## 3. Reversibility PREDICTION accuracy (action-level)

Every action carries `predicted_reversibility` vs `ground_truth_reversibility`. This is the direct measure of whether the model can *classify* reversibility — split by ground-truth class because getting IRREVERSIBLE right is safety-critical.

| Subset | Base acc | Trained acc | Δ | n (base/trained) |
|---|---|---|---|---|
| Overall | 0.8667 | 0.8220 | **-0.0447** | 240/264 |
| On irreversible ops | 0.7438 | 0.6897 | **-0.0541** | 121/145 |
| On reversible ops | 0.9916 | 0.9832 | **-0.0084** | 119/119 |

## 4. Confidence calibration (mean stated confidence)

| | Base | Trained | Δ |
|---|---|---|---|
| Confidence when prediction CORRECT | 0.8780 | 0.8944 | **+0.0164** |
| Confidence when prediction WRONG | 0.8344 | 0.8479 | **+0.0135** |

> Healthy calibration: high confidence when right, LOWER confidence when wrong. If training reduced 'confidence when wrong', the model became better at knowing when it doesn't know.

## 5. Action-type distribution (all steps)

| Action | Base | Trained | Δ |
|---|---|---|---|
| abstain | 235 | 210 | **-25** |
| ask_user | 167 | 186 | **+19** |
| confirm_with_user | 13 | 7 | **-6** |
| execute | 106 | 117 | **+11** |
| respond_to_user | 3 | 3 | **+0** |

