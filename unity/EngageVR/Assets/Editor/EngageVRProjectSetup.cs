// EngageVR -- editor helper that builds the demo scene from script.
//
// The scene is generated rather than checked in as a .unity asset:
// a serialized scene is a binary-ish YAML blob that cannot be reviewed
// in a diff, and the repository policy is to avoid binary assets that
// are not essential. Everything the scene needs is two GameObjects and
// a component reference, which is cheaper to write than to serialize.
//
// Menu: EngageVR > Create Desktop Task Scene

#if UNITY_EDITOR
using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace EngageVR.EditorTools
{
    using EngageVR.Task;
    using EngageVR.UI;

    /// <summary>Creates and saves the desktop task scene.</summary>
    public static class EngageVRProjectSetup
    {
        private const string SceneDirectory = "Assets/Scenes";
        private const string ScenePath = SceneDirectory + "/DesktopTask.unity";

        [MenuItem("EngageVR/Create Desktop Task Scene")]
        public static void CreateDesktopTaskScene()
        {
            Scene scene = EditorSceneManager.NewScene(
                NewSceneSetup.DefaultGameObjects, NewSceneMode.Single);

            GameObject taskObject = new GameObject("TaskController");
            TaskController controller = taskObject.AddComponent<TaskController>();

            // Offline by default: opening the scene and pressing play must
            // not require a Python process to be running.
            controller.useMockTransport = true;

            GameObject hudObject = new GameObject("TaskHud");
            TaskHud hud = hudObject.AddComponent<TaskHud>();
            hud.controller = controller;

            if (!Directory.Exists(SceneDirectory))
            {
                Directory.CreateDirectory(SceneDirectory);
            }

            EditorSceneManager.SaveScene(scene, ScenePath);
            AssetDatabase.Refresh();
            Debug.Log("[EngageVR] created " + ScenePath + " (offline mock transport).");
        }

        [MenuItem("EngageVR/Report Environment")]
        public static void ReportEnvironment()
        {
            Debug.Log(
                "[EngageVR] Unity " + Application.unityVersion
                + " | scripting backend target: " + EditorUserBuildSettings.activeBuildTarget
                + " | protocol " + EngageVR.Protocol.ProtocolVersion.Current);
        }
    }
}
#endif
