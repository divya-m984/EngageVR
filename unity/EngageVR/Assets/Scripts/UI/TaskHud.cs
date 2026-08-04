// EngageVR -- on-screen state display.
//
// Uses IMGUI (OnGUI) rather than uGUI or UI Toolkit so the scene needs
// no prefabs, no serialized canvas hierarchy, and no binary assets. The
// whole HUD is one script, which keeps the repository free of .unity
// and .prefab binaries that cannot be reviewed in a diff.
//
// The HUD reports connection state, block/trial position, the current
// stimulus, and response feedback. It displays no engagement value, no
// cognitive-load value, and no score or performance judgement.

using UnityEngine;

namespace EngageVR.UI
{
    using EngageVR.Networking;
    using EngageVR.Protocol;
    using EngageVR.Task;

    /// <summary>Draws the task state and the stimulus.</summary>
    public sealed class TaskHud : MonoBehaviour
    {
        public TaskController controller;

        public int stimulusSizePixels = 160;

        private GUIStyle _headingStyle;
        private GUIStyle _bodyStyle;
        private Texture2D _stimulusTexture;

        private void Awake()
        {
            if (controller == null)
            {
                controller = FindObjectOfType<TaskController>();
            }

            _stimulusTexture = new Texture2D(1, 1);
            _stimulusTexture.SetPixel(0, 0, Color.white);
            _stimulusTexture.Apply();
        }

        private void OnDestroy()
        {
            if (_stimulusTexture != null)
            {
                Destroy(_stimulusTexture);
            }
        }

        private void OnGUI()
        {
            EnsureStyles();

            if (controller == null)
            {
                GUI.Label(new Rect(16, 16, 600, 24), "No TaskController assigned.", _bodyStyle);
                return;
            }

            DrawStimulus();
            DrawStatusPanel();
        }

        private void EnsureStyles()
        {
            if (_headingStyle != null)
            {
                return;
            }

            _headingStyle = new GUIStyle(GUI.skin.label);
            _headingStyle.fontSize = 18;
            _headingStyle.fontStyle = FontStyle.Bold;

            _bodyStyle = new GUIStyle(GUI.skin.label);
            _bodyStyle.fontSize = 14;
        }

        private void DrawStimulus()
        {
            if (!controller.StimulusVisible || string.IsNullOrEmpty(controller.CurrentStimulusCategory))
            {
                return;
            }

            float size = stimulusSizePixels;
            Rect area = new Rect(
                (Screen.width - size) / 2f,
                (Screen.height - size) / 2f,
                size,
                size);

            Color previous = GUI.color;
            GUI.color = ColorForCategory(controller.CurrentStimulusCategory);
            GUI.DrawTexture(area, _stimulusTexture);
            GUI.color = previous;

            GUI.Label(
                new Rect(area.x, area.yMax + 8, size, 24),
                controller.CurrentStimulusCategory,
                _bodyStyle);
        }

        private void DrawStatusPanel()
        {
            GUILayout.BeginArea(new Rect(16, 16, 520, 260));
            GUILayout.Label("EngageVR desktop reaction task", _headingStyle);

            TransportState state = controller.Transport == null
                ? TransportState.Disconnected
                : controller.Transport.State;
            GUILayout.Label("Connection: " + state + " -- " + controller.ConnectionDetail, _bodyStyle);

            TaskRuntimeState runtime = controller.RuntimeState;
            if (runtime != null)
            {
                GUILayout.Label(
                    "State: " + runtime.State
                    + "   Block: " + Describe(runtime.BlockId)
                    + "   Trial: " + Describe(runtime.TrialId),
                    _bodyStyle);
                GUILayout.Label(
                    "Difficulty: " + runtime.DifficultyLevel
                    + "   Inter-trial interval: " + runtime.StimulusIntervalMs + " ms",
                    _bodyStyle);
            }

            GUILayout.Label(
                "Trials completed: " + controller.CompletedTrialCount
                + "   No response: " + controller.TimeoutCount,
                _bodyStyle);

            if (!string.IsNullOrEmpty(controller.LastFeedback))
            {
                GUILayout.Label("Last trial: " + controller.LastFeedback, _bodyStyle);
            }

            GUILayout.Space(8);
            GUILayout.Label("Press SPACE to start. Respond with J, K, or L.", _bodyStyle);
            GUILayout.Label(
                "Counts above are software telemetry. They are not an engagement, "
                + "attention, cognitive-load, or fatigue measurement.",
                _bodyStyle);
            GUILayout.EndArea();
        }

        private static string Describe(int? value)
        {
            return value.HasValue ? value.Value.ToString() : "-";
        }

        private static Color ColorForCategory(string category)
        {
            switch (category)
            {
                case "square":
                    return new Color(0.31f, 0.53f, 0.85f);
                case "circle":
                    return new Color(0.35f, 0.71f, 0.47f);
                case "triangle":
                    return new Color(0.87f, 0.62f, 0.28f);
                default:
                    return Color.gray;
            }
        }
    }
}
