# AGENTS.md

General-purpose execution rules for coding agents working in a Python 3.12+ software project.

## Operating principles

- If evidence is insufficient, say so. Do not guess.
- Optimize in this order unless the repository says otherwise: correctness, reproducibility, performance, robustness.
- Implement only the requested behavior. Do not add speculative abstractions, backward-compatibility shims, extra configurability, or future-proofing that was not requested.
- Prefer explicit behavior over implicit or automagic behavior.
- Prefer the simplest implementation that satisfies the real requirements and constraints. Do not use clever or hacky solutions.
- This readability-and-correctness bias is the default, not an absolute. If the user explicitly asks for a hot-loop, low-level, or otherwise performance-critical section to be optimized aggressively at the expense of elegance, implement that trade knowingly and with maximum effort, but only for the specifically requested scope.
- Prefer maintained library or framework mechanisms before custom machinery.
- One-time or internal code should still be disciplined and readable, but it should be compact and direct rather than ceremonial.
- Leave the code easier to understand, but do not expand the task into unrelated cleanup.
- Internal code is not exempt from design discipline, but it also does not deserve architecture theater.
- The goal is disciplined, proportionate code: neither sloppy nor pompous.

## Environment and toolchain

- Target Python 3.12+ unless the repository explicitly states otherwise.
- Use the repository environment and lockfile.
- Keep dependency versions pinned in the project configuration.
- Use `ruff`, `ty`, and `pytest` as the default lint, type-check, and test toolchain unless the repository explicitly says otherwise.
- Default command sequence:
  - `ruff check --fix .`
  - `ruff format .`
  - `ruff check --fix .` again because formatting and autofixes can expose follow-up issues for each other
  - `ty check .`
  - `pytest .`
- The Ruff strategy is to enable as many rules as the repository can sustain and iteratively fix the codebase and tests toward compliance.
- Some Ruff rules are mutually exclusive. When conflicts arise, choose the better rule deliberately and document that choice. Do not disable whole classes of rules just to make the problem disappear.
- Keep warning policy centralized instead of scattering warning filters through the codebase.

## Task intake and planning

- Understand the request before changing code.
- State assumptions when they materially affect the implementation.
- If multiple plausible interpretations exist, surface them instead of silently choosing one.
- If a simpler approach exists, say so.
- If clarification materially changes the design, ask. Otherwise make the narrowest reasonable assumption, state it, and proceed.
- For non-trivial work, make a short plan with explicit verification steps.
- Translate vague requests into verifiable success criteria before implementation.
- Generate candidate explanations or solutions, challenge them against the code, tests, logs, and stated contracts, and refine them until the conclusion is stable before acting or answering.
- For broad audits, refactors, or redesigns, start by reading the runbook, onboarding guide, cheat sheet, schema, README, and any equivalent project-entry documents before drawing conclusions.
- Keep the initial reading set narrow and first-party. Do not contaminate a fresh review with unrelated files or hidden context unless the task requires it.

Use this pattern for non-trivial work:

1. Step
   Verify:
2. Step
   Verify:
3. Step
   Verify:

## Scope control and change discipline

- Every changed line should trace directly to the request.
- Touch only the code required for the task.
- Match the surrounding style unless the task is to normalize or refactor it.
- Do not opportunistically reformat, rename, reorganize, or refactor adjacent code.
- Remove imports, variables, functions, files, and tests made obsolete by your own change.
- Do not delete unrelated dead code unless asked.
- Mention unrelated problems you notice, but do not fix them unless they block the task.
- Do not introduce optional knobs, flags, or extension points unless there is a real present need.
- Do not immediately build CLI tools, command wrappers, or user-facing entrypoints unless the task explicitly calls for them.
- If a CLI already exists, do not add new flags, options, switches, or subcommands unless they are explicitly requested or required by an existing contract.
- Strong defaults are better than false flexibility.
- Single-use abstractions need strong justification. If a helper, constant, wrapper, protocol, formatter, shim, or namespace layer is used once and does not improve readability or correctness, inline it.
- Do not replace mess with abstraction theater.
- Do not add random protocols, duck typing layers, namespace indirection, or mocking shenanigans to placate tooling or to imitate architecture.
- Do not preserve backward-compatibility machinery, compatibility tests, or migration scaffolding unless backward compatibility is an explicit requirement.
- Remove files that exist only as empty export stubs or trivial `__all__` declarations unless they provide clear structural value.

## Verification and evidence

- For bugs, reproduce the failure first with a focused test or minimal reproduction, then fix it, then rerun the relevant checks.
- For features, define how success will be verified before implementation.
- For refactors, preserve behavior and run the checks that prove it.
- Work in small, verifiable increments.
- Run the smallest relevant checks first, then widen to broader checks when warranted by risk.
- Do not claim something is verified unless you actually ran the relevant checks.
- When something remains unverified, say exactly what was not checked and why.
- Prefer evidence from code, tests, logs, traces, and docs over speculative root-cause stories.
- Check what downstream code, data consumers, integrations, and dependencies expect, especially at boundaries and for invalid or degenerate inputs.
- When first-party code intentionally accepts imperfect input, verify how that choice affects downstream behavior, quality, performance, training, evaluation, reporting, or user-facing results.
- If a dependency owns a behavior, verify the contract you depend on rather than inventing assumptions about it.

## Design and architecture

- Keep files, modules, classes, and functions cohesive.
- Split long files by real conceptual boundaries. Do not move clutter into generic dump files such as `helpers.py`, `utils.py`, or giant constants or enums modules.
- Do not use modules as loose namespaces for unrelated helpers.
- Keep the package layout shallow and obvious. Do not bury core logic under miscellaneous utility layers.
- Keep helper placement aligned with ownership and domain.
- Keep helpers private unless external use genuinely requires a public surface.
- Separate policy from mechanism, interface from implementation, and workflow choreography from user-facing concepts.
- Avoid deep nesting, long monolithic methods, and repeated multi-step process blocks.
- Extract duplication when the pattern is real and recurring, not merely similar by accident.
- Keep one canonical representation per concept within a layer.
- Do not fossilize temporary migration logic, one-off operational context, or project-history quirks into permanent APIs or core abstractions.
- A reader should be able to guess where code lives before searching for it.
- Architecture must match problem size. Do not build a framework for a small, narrow task.
- Spend complexity only where the domain is actually complex.
- Every abstraction must earn its existence through readability, correctness, reuse, discoverability, or real domain modeling value.
- Keep the number of concepts low. Every new symbol, layer, and file adds tracking cost.
- Prefer repeatable project-wide patterns over a large number of local exceptions.

## Naming and semantics

- Use explicit, precise, semantically aligned names.
- Prefer one canonical name per concept across the codebase.
- Avoid unexplained abbreviations in shared code. Type out symbol names unless the abbreviation is standard and genuinely clearer.
- For example, prefer `annotations` over `anno`.
- Use domain vocabulary accurately and at the correct level of abstraction. Name values by their role, not by a vague container shape.
- If a value mirrors an external field, label, tag, or schema key, use the same name for clear 1:1 correspondence.
- Predicate functions and variables should read like questions: `is_*`, `has_*`, `can_*`, `should_*`.
- Names should encode operational semantics when that matters. Use patterns such as `maybe_*` for helpers that may skip work, `require_*` for getters that raise, and `*_or_*` when a function intentionally has dual behavior.
- Names like `preview_*` are not self-explanatory. State what the function actually does.
- Documentation-only aliases must read as aliases, not as concrete runtime types.
- Do not use names that suggest stronger semantics than the code actually provides.
- Keep terminology stable. Do not use multiple names for the same concept without a real domain reason.

## Types, schemas, and data modeling

- Treat type hints as part of the API contract.
- Use concrete, truthful types. Do not use `Any`, `object`, missing type hints, or fake aliases to make tooling happy.
- Treat the code as if it were strictly typed systems code. Types should provide reasoning support, editor support, and acting documentation.
- Keep type hints aligned with actual runtime usage.
- If something is a schema, use a real schema tool such as Pydantic. If it is structured data but not a schema, prefer dataclasses or `TypedDict`.
- For JSON-like payloads or static-key dictionaries, use structured types with explicit fields and factory helpers instead of anonymous `dict` flows.
- Keep input types as generic as practical and output types as specific as practical.
- Do not misuse `Protocol`, `Literal`, wrapper aliases, or other typing tricks to fake stronger structure than exists.
- If a third-party library has poor typing, contain that awkwardness locally instead of spreading weak typing through the codebase.
- Prefer `type: ignore` with a brief reason over broad or cryptic lint suppression.
- Use enums for closed sets of values. Do not mix enums and raw strings for the same concept.
- Use enum members directly unless conversion is required at an integration boundary.
- Use `Final` for real constants. Do not create extra constants solely to appease lint rules.
- Cast only when the cast is actually necessary.
- Avoid representational drift. Do not model the same concept with multiple unrelated shapes unless there is a deliberate boundary between them.
- Do not fake type safety with random namespace wrappers, loose duck typing, or mocks that erase the real contract.
- Prefer exact interfaces and explicit data flow over vague object-shaped plumbing.

## Python defaults

- Follow the repository formatter, linter, and type checker. If the repo has no explicit style rules, use PEP 8 and Google-style docstrings.
- Prefer `import x` for external packages when it improves clarity. `from x import y` is fine for typing helpers, dataclass helpers, and internal modules when it is the clearer choice.
- Common, standard abbreviations such as `np`, `pd`, and `plt` are allowed when they are the established spelling for the imported package or module. Do not invent ad hoc aliases beyond those broadly recognized conventions.
- When using `__all__`, keep the export surface deliberate and easy to find.
- Do not repeat default argument values in call sites. Pass only the arguments whose values you actually need to change, unless an exact value is contract-critical or spelling it out materially improves readability or intent.
- Prefer the first argument positional and the remaining arguments keyword-only at call sites, unless another calling style is standard and clearly more readable for that API.
- Ambiguous arguments such as booleans, floats, and raw strings should be passed by keyword at call sites.
- When defining functions or methods, make ambiguous parameters such as booleans, floats, and raw strings keyword-only unless positional use is already standard and unambiguous for that API.
- Keep warning policy centralized instead of scattering warning filters through the code.
- Use Pydantic at public constructor or configuration boundaries when it removes boilerplate, but keep it out of repeatedly executed hot-path code unless there is a measured reason.

## Documentation, comments, and strings

- Prefer code that is structurally self-explanatory.
- Fully document modules, classes, functions, methods, and relevant attributes when the repository expects docstrings.
- Keep docstrings consistent, complete, and concise.
- Use Google-style docstrings. Put argument and attribute descriptions on the next indented line.
- Use prose for narrative or contract-heavy docstrings when bullets would make the explanation harder to read.
- Use comments for invariants, math, branching logic, non-obvious design choices, numeric edge cases, device or platform behavior, replay boundaries, and other contract-sensitive behavior.
- Do not write comments that merely restate obvious code.
- Document workarounds with why they exist, when they matter, and when they can be removed.
- Prefer explicit names, typed signatures, and small helper boundaries over explanatory comments.
- Use whitespace and blank lines to separate distinct phases of logic.
- Do not fragment message strings into awkward pieces. Prefer full templates or small formatting helpers when the message is genuinely reused.
- Keep comments, docstrings, logs, warnings, and errors concise, technical, and consistent in tone.
- Do not be needlessly verbose, but do not leave important behavior to chance or reader guesswork.
- Keep the docs, runbook, onboarding, cheat sheet, schema, README, code, and tests aligned with each other. When one changes, update the others if the contract moved.

## Error handling, contracts, and configuration

- Fail fast on unsupported setups instead of adding soft fallbacks unless graceful degradation is part of the requirement.
- Do not add defensive error handling for impossible scenarios.
- Be consistent about custom errors. Do not mix many one-off local error styles without reason.
- Avoid single-use error formatter helpers.
- Make boundaries and invariants explicit in code and tests.
- Prefer deterministic, opinionated defaults over sprawling configuration surfaces.
- Expose configurability only when there is a real operational use case.
- Treat changes to default behavior as contract-sensitive when they affect compatibility, performance, reproducibility, or operator expectations.
- Human-facing reports should summarize totals and aggregate counts when readers would otherwise need to mentally add detail rows to understand the result.
- State contracts at boundaries and verify them there. Do not scatter the same validation everywhere without reason.

## Math and numerics

- For floating-point equality checks, use `math.isclose` with the default relative and absolute tolerances unless there is a specific, documented reason to override them.
- Use ordinary `<`, `>`, `<=`, and `>=` comparisons when ordering is the real requirement. Do not replace normal inequality checks with tolerance logic out of habit.
- Introduce epsilon values only to prevent real divide-by-zero or similarly unavoidable numeric singularities, not as a generic patch for unclear comparisons or unstable logic.
- Prefer textbook-like implementations of mathematical logic. Make the computation easy to map to the underlying formula, derivation, or invariant.
- Prioritize mathematical correctness and readability over cleverness, compression, or trick implementations. Make extra effort not to be smart when writing math-heavy code.
- Use real variable names in mathematical code. Do not fall back to meaningless one-letter names unless the symbol is standard, local, and genuinely clarifying.
- Add concise comments for non-obvious derivations, invariants, coordinate conventions, units, edge cases, or stability decisions.

## Performance and dependencies

- Keep hot paths hot. Do not add repeated validation, manifest bookkeeping, payload audits, or similar defensive work to frequently executed paths without a demonstrated need.
- Prefer readability over micro-optimizations unless performance work is requested or measured evidence shows a bottleneck.
- When the user explicitly asks for maximum performance in a specific path, prioritize performance in that path even if the resulting code is less elegant, provided the code remains correct, well-bounded, and documented enough to explain the trade.
- Benchmark performance-sensitive changes incrementally. Do not assume a technique helps simply because it is commonly recommended.
- Avoid unnecessary copies, conversions, and repeated work in data-heavy code.
- Keep runtime logic platform- and device-agnostic where possible.
- Do not broaden the dependency surface without a clear reason.
- When changing performance-sensitive defaults, justify the change with measurements.
- Check whether downstream consumers tolerate invalid, incomplete, or geometrically degenerate input before deciding to pass it through, repair it, or reject it.
- Make that decision based on downstream impact, not on convenience alone.

## Tests

- Treat tests as first-class code. Apply the same instruction set, standards, constraints, and quality bar to the test suite as to production code unless a rule is explicitly limited to production-only behavior. Naming, typing, structure, documentation, simplicity, and review expectations all apply equally to tests.
- In tests, import the codebase through its fully qualified package paths, as if it were a third-party library, so production imports are visually distinct from test-local code and helpers.
- Write or update tests alongside behavior changes.
- Add focused regression tests for bugs and behavior changes.
- Prefer tests that explain behavior and contracts over tests that mirror implementation details.
- Treat tests as documentation in code form.
- Test first-party contracts, not third-party contracts, unless your code wraps, extends, or depends on the exact third-party behavior.
- Do not spend test coverage re-proving validation or defaults owned entirely by dependencies unless your code makes that behavior part of your own contract.
- Run the smallest relevant test target first, then widen to the affected module, then the full suite as needed.
- Use deterministic seeds, fixtures, builders, and temporary directories.
- Prefer small fake data, builders, or fixture files over bulky inline blobs.
- Use parametrization when it makes tests easier to understand than repetitive walls of assertions.
- Assert warnings and exceptions intentionally.
- Mock or isolate external dependencies only when that preserves the real contract under test. Do not use mocks to blur or replace important type and behavior boundaries.
- For numerical or stateful systems, prefer asserting state transitions, invariants, ordering, checkpoint semantics, data properties, and mode changes over brittle exact floating-point values unless determinism is guaranteed.
- Delete obsolete tests that exist only for removed compatibility layers, removed defenses, or implementation details that no longer define behavior.
- Keep the test suite aligned with the codebase, and keep both aligned with the runbook, onboarding, cheat sheet, schema, README, and other first-party docs.

## Communication and completion

- Be concise, neutral, and code-focused.
- Surface uncertainty, assumptions, and tradeoffs early.
- If you choose between multiple valid options, explain the reason briefly.
- In summaries and reviews, explain why the change is correct, note contract-sensitive areas, and list the checks you ran.
- Separate optional follow-up cleanup from the requested work.
- Say exactly what remains unverified.
- The goal is code that another engineer can understand, trust, and modify without needing the original author in the room.
