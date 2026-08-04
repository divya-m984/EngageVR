# Unity Desktop Task — Setup and Status

## Validation status

> **Unity compilation and runtime validation are PENDING.**
>
> No Unity Editor and no Unity Hub is installed in the environment where this
> code was written (`unity-editor` and `unityhub` are not on `PATH`; there is
> no `~/Unity`, `/opt/unity*`, or `/usr/bin/unity*`). Unity was **not**
> downloaded or installed automatically.
>
> Consequently:
> - the C# code has **not been compiled**;
> - the EditMode and PlayMode tests have **not been executed**;
> - no player has been built;
> - the Unity acceptance criteria are **not** marked as passed.
>
> What *has* been validated is the protocol contract the C# code is written
> against: the checked-in fixtures under `protocol/fixtures/` are parsed and
> asserted by the Python test suite, and the Unity EditMode tests parse those
> same files.

## What exists

```
unity/EngageVR/
  .gitignore                       Library/, Temp/, Logs/, Builds/, UserSettings/, ...
  Packages/manifest.json
  Assets/
    Scripts/
      EngageVR.Runtime.asmdef
      Protocol/
        Json.cs                    dependency-free JSON with first-class null
        ProtocolEnvelope.cs        envelope, versioning, provenance, replay metadata
        ProtocolMessages.cs        typed payloads
      Networking/
        IMessageTransport.cs       transport abstraction
        WebSocketTransport.cs      System.Net.WebSockets.ClientWebSocket
        MockTransport.cs           offline; answers the handshake
      Task/
        TaskController.cs          the MonoBehaviour loop
        TrialController.cs         trial plan and one trial's lifecycle
        TaskTelemetry.cs           builds task-event messages
        AdaptationReceiver.cs      applies commands, builds acknowledgements
      UI/
        TaskHud.cs                 IMGUI state display
    Editor/
      EngageVR.Editor.asmdef
      EngageVRProjectSetup.cs      menu: EngageVR > Create Desktop Task Scene
    Tests/
      EditMode/  EngageVR.Tests.EditMode.asmdef, ProtocolContractTests.cs, TaskLogicTests.cs
      PlayMode/  EngageVR.Tests.PlayMode.asmdef, TaskControllerPlayModeTests.cs
```

`ProjectSettings/` is deliberately **not** checked in with a fabricated
`ProjectVersion.txt`: this repository cannot verify which Unity version the
project opens under, and asserting one would be a claim it has not tested.
Unity Hub will ask which version to use on first open.

## Opening the project

1. Install Unity Hub and a currently supported **Unity LTS** release
   (Unity 6 LTS or 2022.3 LTS; both provide the .NET Standard 2.1 profile
   this code targets).
2. In Unity Hub: **Add** → select `unity/EngageVR`.
3. Open it. Unity generates `Library/`, `ProjectSettings/`, and the solution
   files; all are gitignored.
4. Menu → **EngageVR → Create Desktop Task Scene**. This writes
   `Assets/Scenes/DesktopTask.unity` with a `TaskController` and a `TaskHud`,
   configured for the **offline mock transport**.
5. Press Play. Press **Space** to start; respond with **J**, **K**, **L**.

No headset, no OpenXR, no XR Interaction Toolkit, and no VR hardware is
required or referenced. No paid assets. No binary assets: the HUD is IMGUI
and the scene is generated from script rather than checked in as a serialized
`.unity` blob that cannot be reviewed in a diff.

## Running the tests

Window → General → **Test Runner**, then run the EditMode and PlayMode tabs.

Batch mode, once an Editor is installed (record the exact version used):

```bash
<UnityEditor> -runTests -batchmode \
  -projectPath unity/EngageVR \
  -testPlatform EditMode \
  -testResults artifacts/unity-editmode-results.xml \
  -logFile artifacts/unity-editmode.log

<UnityEditor> -runTests -batchmode \
  -projectPath unity/EngageVR \
  -testPlatform PlayMode \
  -testResults artifacts/unity-playmode-results.xml \
  -logFile artifacts/unity-playmode.log
```

A Linux development player, if licensing and environment permit:

```bash
<UnityEditor> -quit -batchmode \
  -projectPath unity/EngageVR \
  -buildLinux64Player artifacts/unity-build/EngageVR.x86_64 \
  -logFile artifacts/unity-build.log
```

**None of these commands has been run.** They are instructions, not a record.

## Connecting to the backend

On the `TaskController` component:

- `Use Mock Transport` **on** (default) — runs entirely offline.
- `Use Mock Transport` **off** — set `Web Socket Url` to
  `ws://127.0.0.1:8000/ws/v1/sessions/unity-session` and start the backend:

```bash
uv run python -m engagevr serve --host 127.0.0.1 --port 8000
```

Loopback only. The backend has no authentication.

## Design decisions

### No `JsonUtility` — a correctness requirement

Unity's built-in `JsonUtility` **cannot represent `null`**. It serializes a
null string as `""` and a nullable number as `0`, and on deserialization it
leaves absent fields at their default. The EngageVR protocol depends on the
difference between "no response" (`null`) and "a response of 0 ms", and
between "no reaction time" and "0 ms". Using `JsonUtility` would silently
convert every missed trial into a zero-latency response — exactly what the
Python schema forbids. It also cannot serialize dictionaries or top-level
arrays, both of which the protocol uses.

`Assets/Scripts/Protocol/Json.cs` is therefore a small, dependency-free JSON
reader/writer in which `null` is a first-class kind. It refuses to serialize
`NaN` or `Infinity` rather than substituting a placeholder, since JSON cannot
represent them and a substitute would be fabricated data.

Newtonsoft (`com.unity.nuget.newtonsoft-json`) would also have worked, but
adding a package dependency that has not been resolved or compiled in this
repository would be an unverified claim.

### `System.Net.WebSockets.ClientWebSocket` — no third-party package

`ClientWebSocket` ships with the .NET Standard 2.1 base class library that
Unity's Mono and IL2CPP backends expose. It needs no package, no manifest
entry, and no licence review, and it works in the Editor and in
Windows/macOS/Linux standalone players.

**Documented limitation:** `ClientWebSocket` is *not* supported on the WebGL
player, where the browser owns the socket. EngageVR targets a desktop player,
so this is not a constraint here; a WebGL build would need a JavaScript
interop bridge and is out of scope.

No unverified third-party WebSocket package was added.

Both decisions are recorded in `docs/DECISIONS.md` (DEC-030, DEC-031).

### Threading

`WebSocketTransport` sends and receives on background tasks and touches no
Unity API. Inbound messages are queued and dispatched on the main thread by
`Poll()`, called once per frame from `Update`, because Unity API calls from a
background thread are undefined behaviour.

## Protocol contract

The C# field names are the wire contract and must match the Python models
exactly. Both test suites parse the same checked-in fixtures under
`protocol/fixtures/`, so a rename on either side fails the other side's
tests. `ProtocolContractTests.GeneratedTaskEventMatchesTheFixtureFieldSet`
compares the full C# task-event field set against the Python one.

Regenerate the fixtures after any protocol change:

```bash
uv run python scripts/generate_protocol_artifacts.py
```

Note: `TrialPlanBuilder` reproduces the Python generator's *structure*, not
its exact draws — C# and Python use different RNG algorithms, so the two
sequences are each internally deterministic but are not identical to one
another. The protocol contract is on the message format, not on stimulus
order.

## Scope

The Unity task is a **software telemetry source**. Its accuracy, reaction
times, and timeout counts are not engagement, attention, cognitive-load, or
fatigue measurements. The task has not been experimentally designed, piloted,
or approved.

The client implements only visually harmless commands — `set_difficulty`,
`set_stimulus_interval`, `pause_task`, `resume_task` — and makes **no
automatic adaptation decisions**. A PlayMode test asserts that no message the
client emits contains the words `engagement`, `cognitive_load`, `attention`,
or `fatigue`.
