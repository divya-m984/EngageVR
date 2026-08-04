// EngageVR -- the trial plan and one trial's lifecycle.
//
// Kept free of UnityEngine types so EditMode tests can drive a whole
// task without entering play mode.
//
// The stimulus vocabulary matches engagevr/task/generator.py: three
// abstract shapes with one response key each. Deliberately abstract:
// no semantic, emotional, or clinical content.

using System;
using System.Collections.Generic;

namespace EngageVR.Task
{
    /// <summary>One trial's definition, fixed before it runs.</summary>
    public sealed class TrialDefinition
    {
        public int BlockId;
        public int TrialId;
        public string StimulusId;
        public string StimulusCategory;
        public string ExpectedResponse;
    }

    /// <summary>How a trial's response slot resolved.</summary>
    public enum TrialResolution
    {
        Pending,
        Responded,
        TimedOut
    }

    /// <summary>Builds a deterministic trial plan from a seed.</summary>
    public static class TrialPlanBuilder
    {
        /// <summary>Stimulus categories, matching the Python generator.</summary>
        public static readonly string[] StimulusCategories = { "square", "circle", "triangle" };

        /// <summary>Response keys, one per category, in the same order.</summary>
        public static readonly string[] ResponseKeys = { "j", "k", "l" };

        /// <summary>
        /// Build the plan. The stimulus sequence is seeded, so the same
        /// seed produces the same sequence on every run.
        /// <para>
        /// Note: this reproduces the Python generator's *structure*, not
        /// its exact draws — C# and Python use different RNG algorithms,
        /// so the two sequences are each internally deterministic but are
        /// not identical to one another. The protocol contract is on the
        /// message format, not on the stimulus order.
        /// </para>
        /// </summary>
        public static List<TrialDefinition> Build(int blocks, int trialsPerBlock, int seed)
        {
            if (blocks < 1)
            {
                throw new ArgumentOutOfRangeException("blocks", "blocks must be at least 1");
            }

            if (trialsPerBlock < 1)
            {
                throw new ArgumentOutOfRangeException(
                    "trialsPerBlock", "trialsPerBlock must be at least 1");
            }

            Random random = new Random(seed);
            List<TrialDefinition> plan = new List<TrialDefinition>(blocks * trialsPerBlock);

            for (int blockId = 0; blockId < blocks; blockId++)
            {
                for (int trialId = 0; trialId < trialsPerBlock; trialId++)
                {
                    int index = random.Next(StimulusCategories.Length);
                    string category = StimulusCategories[index];
                    plan.Add(new TrialDefinition
                    {
                        BlockId = blockId,
                        TrialId = trialId,
                        StimulusCategory = category,
                        StimulusId = category + "-b" + blockId + "t" + trialId,
                        ExpectedResponse = ResponseKeys[index]
                    });
                }
            }

            return plan;
        }
    }

    /// <summary>
    /// Tracks one trial from stimulus onset to resolution.
    /// <para>
    /// Time is supplied by the caller rather than read from
    /// <c>UnityEngine.Time</c>, so a test can advance it directly.
    /// </para>
    /// </summary>
    public sealed class TrialController
    {
        private readonly double _responseTimeoutMs;
        private double _stimulusOnsetMs;
        private bool _stimulusVisible;

        public TrialController(double responseTimeoutMs)
        {
            if (responseTimeoutMs <= 0.0)
            {
                throw new ArgumentOutOfRangeException(
                    "responseTimeoutMs", "the response timeout must be positive");
            }

            _responseTimeoutMs = responseTimeoutMs;
        }

        public TrialDefinition Current { get; private set; }

        public TrialResolution Resolution { get; private set; }

        /// <summary>Reaction time, or null when no response was registered.</summary>
        public double? ReactionTimeMs { get; private set; }

        /// <summary>Observed key, or null when no response was registered.</summary>
        public string ObservedResponse { get; private set; }

        /// <summary>Correctness, or null when no response was registered.</summary>
        public bool? ResponseCorrect { get; private set; }

        public bool StimulusVisible
        {
            get { return _stimulusVisible; }
        }

        /// <summary>Begin a trial and present its stimulus at <paramref name="nowMs"/>.</summary>
        public void Begin(TrialDefinition definition, double nowMs)
        {
            if (definition == null)
            {
                throw new ArgumentNullException("definition");
            }

            Current = definition;
            Resolution = TrialResolution.Pending;
            ReactionTimeMs = null;
            ObservedResponse = null;
            ResponseCorrect = null;
            _stimulusOnsetMs = nowMs;
            _stimulusVisible = true;
        }

        /// <summary>
        /// Register a key press. Ignored unless the trial is pending.
        /// Returns true when this press resolved the trial.
        /// </summary>
        public bool RegisterResponse(string key, double nowMs)
        {
            if (Resolution != TrialResolution.Pending || Current == null)
            {
                return false;
            }

            double reaction = nowMs - _stimulusOnsetMs;
            if (reaction < 0.0)
            {
                // A negative reaction time is impossible; clamping to
                // zero would fabricate a plausible value, so the press
                // is refused instead.
                return false;
            }

            ObservedResponse = key;
            ReactionTimeMs = reaction;
            ResponseCorrect = string.Equals(
                key, Current.ExpectedResponse, StringComparison.Ordinal);
            Resolution = TrialResolution.Responded;
            _stimulusVisible = false;
            return true;
        }

        /// <summary>
        /// Advance the clock. Returns true when this tick timed the
        /// trial out. A timeout leaves reaction time and correctness
        /// null: no response is not a zero-millisecond wrong response.
        /// </summary>
        public bool Tick(double nowMs)
        {
            if (Resolution != TrialResolution.Pending || Current == null)
            {
                return false;
            }

            if (nowMs - _stimulusOnsetMs < _responseTimeoutMs)
            {
                return false;
            }

            Resolution = TrialResolution.TimedOut;
            ObservedResponse = null;
            ReactionTimeMs = null;
            ResponseCorrect = null;
            _stimulusVisible = false;
            return true;
        }

        public double ElapsedSinceStimulusMs(double nowMs)
        {
            return Math.Max(nowMs - _stimulusOnsetMs, 0.0);
        }
    }
}
