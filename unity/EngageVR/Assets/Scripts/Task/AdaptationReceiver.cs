// EngageVR -- applies adaptation commands and builds acknowledgements.
//
// MILESTONE 4 BOUNDARY
// --------------------
// This class APPLIES commands that arrive. It never DECIDES them.
// There is no policy, no cooldown, no hysteresis, and no derivation of
// a command from task performance anywhere in this client. Nothing here
// claims that applying a command improves engagement or any other
// outcome.
//
// The accept/reject rules mirror engagevr/task/state.py exactly, so the
// Unity client and the Python simulator behave identically for the same
// command in the same state.

using System;
using System.Collections.Generic;

namespace EngageVR.Task
{
    using EngageVR.Protocol;

    /// <summary>The mutable settings a command may change.</summary>
    public sealed class TaskRuntimeState
    {
        public string State = TaskStateName.Idle;
        public int DifficultyLevel = 1;
        public double StimulusIntervalMs = 500.0;
        public int? BlockId;
        public int? TrialId;
    }

    /// <summary>Receives adaptation commands and answers each one.</summary>
    public sealed class AdaptationReceiver
    {
        private readonly TaskRuntimeState _state;
        private readonly Dictionary<string, string> _applied =
            new Dictionary<string, string>(StringComparer.Ordinal);

        public AdaptationReceiver(TaskRuntimeState state)
        {
            if (state == null)
            {
                throw new ArgumentNullException("state");
            }

            _state = state;
        }

        /// <summary>How many distinct commands have been applied.</summary>
        public int AppliedCount
        {
            get { return _applied.Count; }
        }

        /// <summary>
        /// Apply one command and build its acknowledgement.
        /// <para>
        /// A repeated command id is acknowledged again with
        /// <c>Duplicate = true</c> and is NOT re-applied, so a
        /// retransmitted command cannot double-step the difficulty.
        /// </para>
        /// </summary>
        public AdaptationAcknowledgementPayload Apply(
            AdaptationCommandPayload command, DateTime nowUtc)
        {
            if (command == null)
            {
                throw new ArgumentNullException("command");
            }

            string previouslyAppliedAt;
            if (_applied.TryGetValue(command.CommandId, out previouslyAppliedAt))
            {
                return new AdaptationAcknowledgementPayload
                {
                    CommandId = command.CommandId,
                    Accepted = true,
                    AppliedAtUtc = previouslyAppliedAt,
                    Duplicate = true
                };
            }

            if (command.IsExpired(nowUtc))
            {
                return Reject(
                    command,
                    "command expired at " + command.ExpiresAtUtc + "; it reached the "
                    + "client at " + ProtocolEnvelope.FormatUtc(nowUtc));
            }

            switch (command.Command)
            {
                case AdaptationCommandName.SetDifficulty:
                {
                    if (command.Value == null || command.Value.Kind != JsonKind.Number)
                    {
                        return Reject(command, "set_difficulty requires a numeric value");
                    }

                    int difficulty = (int)command.Value.AsNumber;
                    if (difficulty < 0)
                    {
                        return Reject(command, "set_difficulty value must be non-negative");
                    }

                    _state.DifficultyLevel = difficulty;
                    break;
                }

                case AdaptationCommandName.SetStimulusInterval:
                {
                    if (command.Value == null || command.Value.Kind != JsonKind.Number)
                    {
                        return Reject(
                            command, "set_stimulus_interval requires a numeric value");
                    }

                    double interval = command.Value.AsNumber;
                    if (interval <= 0.0)
                    {
                        return Reject(
                            command, "set_stimulus_interval value must be positive");
                    }

                    _state.StimulusIntervalMs = interval;
                    break;
                }

                case AdaptationCommandName.PauseTask:
                {
                    if (!string.Equals(
                            _state.State, TaskStateName.Running, StringComparison.Ordinal))
                    {
                        return Reject(
                            command,
                            "pause_task requires state 'running'; the task is '"
                            + _state.State + "'");
                    }

                    _state.State = TaskStateName.Paused;
                    break;
                }

                case AdaptationCommandName.ResumeTask:
                {
                    if (!string.Equals(
                            _state.State, TaskStateName.Paused, StringComparison.Ordinal))
                    {
                        return Reject(
                            command,
                            "resume_task requires state 'paused'; the task is '"
                            + _state.State + "'");
                    }

                    _state.State = TaskStateName.Running;
                    break;
                }

                default:
                    return Reject(
                        command,
                        "this client implements only set_difficulty, "
                        + "set_stimulus_interval, pause_task, and resume_task; got '"
                        + command.Command + "'");
            }

            string appliedAt = ProtocolEnvelope.FormatUtc(nowUtc);
            _applied[command.CommandId] = appliedAt;
            return new AdaptationAcknowledgementPayload
            {
                CommandId = command.CommandId,
                Accepted = true,
                AppliedAtUtc = appliedAt,
                Duplicate = false
            };
        }

        private static AdaptationAcknowledgementPayload Reject(
            AdaptationCommandPayload command, string reason)
        {
            return new AdaptationAcknowledgementPayload
            {
                CommandId = command.CommandId,
                Accepted = false,
                AppliedAtUtc = null,
                RejectionReason = reason,
                Duplicate = false
            };
        }
    }
}
