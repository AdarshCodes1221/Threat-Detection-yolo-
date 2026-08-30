# Ethics and Limitations

## Interpretation Boundary

The model recognizes visual patterns defined by the training labels. It cannot determine a person's identity, intent, legal status, mental state, or future behavior. The phrase `potentially_dangerous_person` is therefore a narrow dataset label, not a statement about the person.

## Human Oversight

Every alert requires contextual review by trained personnel. No arrest, denial of access, accusation, or other adverse action should be based on the model output alone. Organizations should define responsibility, escalation, appeal, and incident-audit procedures before any trial.

## Error Costs

- False negatives may fail to surface a relevant scene.
- False positives may stigmatize or inconvenience an innocent person.
- Confidence scores are not probabilities of harmful intent.
- Repeated alerts can create automation bias and alert fatigue.

## Bias and Generalization

Results may vary with skin tone, clothing, mobility aids, cultural context, image compression, camera height, lighting, weather, crowd density, distance, object size, and occlusion. The supplied evaluation is not a fairness audit and does not justify use on an untested population.

## Privacy and Governance

Public-area imagery can still contain personal data. Any applied study should establish a lawful purpose, proportionality, notice where required, retention limits, access control, secure storage, audit logging, deletion procedures, and independent review.

## Prohibited Uses

- Face recognition or biometric identification.
- Demographic, emotion, personality, or criminality inference.
- Predictive policing or generalized population surveillance.
- Autonomous use of force or punitive decision-making.
- Concealed deployment that violates applicable law or policy.

## Recommended Validation

Before a controlled field study, use representative local data and report detection quality, calibration, latency, threshold sensitivity, subgroup behavior, failure modes, privacy controls, and human-factors testing. Stop the study if harms cannot be acceptably reduced.
