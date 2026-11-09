# Soma Intake Evaluation

_50 synthetic submissions, 4.7s, deterministic extractor (no LLM)._

## Headline metrics

| metric | value |
|---|---|
| Extraction F1 (micro, 24 ACORD fields) | 99.8%  (P 100.0% / R 99.6%) |
| Portal-fill success rate | 100.0%  (38/38 cases that should submit) |
| Average turns to completion | 5.08 (min 5, max 6) |
| Human-handoff precision | 100.0%  (12/12) |
| Human-handoff recall | 100.0%  (12/12) |
| Process restarts survived | 150 |
| Replay determinism (fold(history) == live state) | True |
| Accuracy when confidence >= 0.80 | 1.0 (n=1155) |
| Accuracy when confidence < 0.80 | 1.0 (n=8) |

## Outcomes by scenario

| scenario | cases | submitted | handed off |
|---|---|---|---|
| clean | 18 | 18 | 0 |
| needs_followup | 14 | 14 | 0 |
| conflict_corrected | 6 | 6 | 0 |
| high_risk | 7 | 0 | 7 |
| ambiguous | 5 | 0 | 5 |

## Crash recovery (process killed between activity_started and activity_completed)

| case | crashed in | recovered | final status | portal submissions | outbound msgs | follow-up activities | exactly once |
|---|---|---|---|---|---|---|---|
| 1 | portal | True | SUBMITTED | 1 | 1 | 1 | True |
| 2 | followup | True | SUBMITTED | 1 | 2 | 2 | True |
| 3 | portal | True | SUBMITTED | 1 | 1 | 1 | True |
| 4 | followup | True | SUBMITTED | 1 | 2 | 2 | True |
| 5 | portal | True | SUBMITTED | 1 | 1 | 1 | True |
| 6 | followup | True | SUBMITTED | 1 | 2 | 2 | True |

## Per-field extraction accuracy

| field | ACORD | required | correct / present |
|---|---|---|---|
| applicant_name | 125 | True | 50/50 |
| entity_type | 125 | True | 50/50 |
| fein | 125 | False | 50/50 |
| mailing_street | 125 | True | 50/50 |
| mailing_city | 125 | True | 50/50 |
| mailing_state | 125 | True | 50/50 |
| mailing_zip | 125 | True | 50/50 |
| business_phone | 125 | False | 50/50 |
| contact_name | 125 | True | 50/50 |
| contact_email | 125 | True | 50/50 |
| years_in_business | 125 | False | 50/50 |
| business_description | 125 | True | 50/50 |
| naics | 125 | False | 50/50 |
| employees_full_time | 125 | True | 50/50 |
| employees_part_time | 125 | False | 50/50 |
| annual_revenue | 125 | True | 45/50 |
| policy_effective_date | 125 | True | 50/50 |
| lines_of_business | 125 | True | 50/50 |
| gl_each_occurrence | 126 | True | 50/50 |
| gl_general_aggregate | 126 | True | 50/50 |
| gl_deductible | 126 | False | 50/50 |
| prior_losses_3y | 126 | True | 50/50 |
| prior_carrier | 125 | True | 50/50 |
| hazards | 126 | False | 18/18 |

## Handoffs

| case | scenario | reasons |
|---|---|---|
| 8 | high_risk | high_risk_hazard:explosives |
| 14 | ambiguous | low_confidence:annual_revenue=0.00 |
| 17 | ambiguous | low_confidence:annual_revenue=0.00 |
| 22 | high_risk | high_risk_hazard:liquor |
| 28 | high_risk | high_risk_hazard:roof_work |
| 30 | ambiguous | low_confidence:annual_revenue=0.00 |
| 32 | ambiguous | low_confidence:annual_revenue=0.00 |
| 38 | high_risk | high_risk_hazard:explosives |
| 42 | high_risk | high_risk_hazard:explosives |
| 43 | high_risk | high_risk_hazard:explosives |
| 46 | high_risk | high_risk_hazard:roof_work |
| 47 | ambiguous | low_confidence:annual_revenue=0.00 |

Harness: 204 workflows, 310 tool calls, 0 guard blocks.

## Generalisation check: held-out phrasing

Same 50 businesses and scenarios, but every customer statement uses wording that was NEVER used while writing the extraction rules (e.g. 'per claim', 'yearly turnover', 'zero losses', 'replace shingles'). Rules were not tuned on this set.

| metric | development phrasing | held-out phrasing |
|---|---|---|
| Extraction F1 | 99.8% | 91.7% |
| Extraction precision / recall | 100.0% / 99.6% | 95.2% / 88.5% |
| Portal-fill success | 100.0% (38/38) | 21.1% (8/38) |
| Avg turns to completion | 5.08 | 5.08 |
| Handoff precision | 100.0% (12/12) | 32.4% (12/37) |
| Handoff recall | 100.0% (12/12) | 100.0% (12/12) |
| Accuracy at confidence >= 0.80 | 1.0 | 0.9661 |

Held-out per-field accuracy (fields below 100%):

| field | correct / present |
|---|---|
| years_in_business | 11/50 |
| employees_full_time | 26/50 |
| employees_part_time | 31/50 |
| annual_revenue | 40/50 |
| gl_each_occurrence | 39/50 |
| gl_general_aggregate | 39/50 |
| gl_deductible | 39/50 |
| prior_carrier | 42/50 |
| hazards | 15/16 |

Held-out cases that should have been submitted but were not (or with wrong values):

- case 1 (conflict_corrected): wrong values ['employees_full_time']
- case 2 (needs_followup): status HUMAN_HANDOFF, reasons ['unresolved_conflict:annual_revenue=0.44']
- case 5 (needs_followup): status HUMAN_HANDOFF, reasons ['unresolved_conflict:annual_revenue=0.49,gl_each_occurrence=0.00,gl_general_aggregate=0.00']
- case 6 (clean): status HUMAN_HANDOFF, reasons ['low_confidence:gl_each_occurrence=0.00,gl_general_aggregate=0.00']
- case 9 (needs_followup): wrong values ['employees_full_time']
- case 10 (clean): status HUMAN_HANDOFF, reasons ['unresolved_conflict:annual_revenue=0.49']
- case 12 (needs_followup): status HUMAN_HANDOFF, reasons ['unresolved_conflict:annual_revenue=0.49']
- case 13 (clean): status HUMAN_HANDOFF, reasons ['unresolved_conflict:annual_revenue=0.49,gl_each_occurrence=0.00,gl_general_aggregate=0.00']
- case 15 (clean): status HUMAN_HANDOFF, reasons ['unresolved_conflict:annual_revenue=0.49,gl_each_occurrence=0.00,gl_general_aggregate=0.00']
- case 18 (needs_followup): status HUMAN_HANDOFF, reasons ['low_confidence:prior_carrier=0.00']
- case 19 (clean): status HUMAN_HANDOFF, reasons ['unresolved_conflict:annual_revenue=0.49']
- case 20 (clean): status HUMAN_HANDOFF, reasons ['low_confidence:prior_carrier=0.00']
- case 21 (clean): wrong values ['employees_full_time']
- case 23 (clean): status HUMAN_HANDOFF, reasons ['unresolved_conflict:annual_revenue=0.49']
- case 24 (needs_followup): status HUMAN_HANDOFF, reasons ['unresolved_conflict:annual_revenue=0.49,gl_each_occurrence=0.00,gl_general_aggregate=0.00']
- case 25 (conflict_corrected): status HUMAN_HANDOFF, reasons ['unresolved_conflict:annual_revenue=0.49']
- case 26 (needs_followup): status HUMAN_HANDOFF, reasons ['unresolved_conflict:annual_revenue=0.49,gl_each_occurrence=0.00,gl_general_aggregate=0.00']
- case 27 (conflict_corrected): wrong values ['employees_full_time']
- case 29 (clean): status HUMAN_HANDOFF, reasons ['unresolved_conflict:annual_revenue=0.49']
- case 33 (clean): status HUMAN_HANDOFF, reasons ['low_confidence:prior_carrier=0.00']
- case 34 (clean): status HUMAN_HANDOFF, reasons ['unresolved_conflict:annual_revenue=0.49']
- case 35 (clean): status HUMAN_HANDOFF, reasons ['unresolved_conflict:annual_revenue=0.49']
- case 36 (needs_followup): wrong values ['employees_full_time']
- case 39 (conflict_corrected): status HUMAN_HANDOFF, reasons ['unresolved_conflict:annual_revenue=0.49,gl_each_occurrence=0.00,gl_general_aggregate=0.00']
- case 40 (needs_followup): status HUMAN_HANDOFF, reasons ['unresolved_conflict:annual_revenue=0.49,gl_each_occurrence=0.00,gl_general_aggregate=0.00']
- case 41 (conflict_corrected): status HUMAN_HANDOFF, reasons ['unresolved_conflict:annual_revenue=0.49']
- case 44 (clean): status HUMAN_HANDOFF, reasons ['unresolved_conflict:annual_revenue=0.49,gl_each_occurrence=0.00,gl_general_aggregate=0.00,prior_carrier=0.00']
- case 48 (conflict_corrected): status HUMAN_HANDOFF, reasons ['unresolved_conflict:annual_revenue=0.49']
- case 49 (needs_followup): status HUMAN_HANDOFF, reasons ['unresolved_conflict:annual_revenue=0.49,gl_each_occurrence=0.00,gl_general_aggregate=0.00']
- case 50 (clean): status HUMAN_HANDOFF, reasons ['unresolved_conflict:annual_revenue=0.49,gl_each_occurrence=0.00,gl_general_aggregate=0.00']

Held-out high-risk / ambiguous cases handed off: 12/12. Any miss here is a high-risk business that reached the carrier portal without underwriter review, which is the most important failure mode to close before production (see the note at the top).
