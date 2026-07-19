# Frozen ScenarioNet Live Content Validation

Status: `PASS`

## Scope

All checks opened immutable source files read-only. No source file, catalog, 
ScenarioDescription, split, tag, or route annotation was changed.

## Results

- Records checked: `3500`
- Outcome counts: `{'pass': 3500}`
- Structural checks: source file existence, pickle loadability, mapping shape, 
catalog/description horizon equality, nonempty tracks/map, SDC state, position/heading 
finite values and lengths, nonempty SDC valid mask, and assigned-route lane membership.
- Not evaluated: simulator reset/step, Rulebook runtime eligibility, controls semantics, 
policy termination/truncation, and learning behavior.
