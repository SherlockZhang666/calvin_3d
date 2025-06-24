import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from omegaconf import OmegaConf
import hydra

from calvin_env.envs.play_table_env import get_env
from calvin_agent.evaluation.utils import get_env_state_for_initial_condition
from calvin_agent.evaluation.multistep_sequences import get_sequences

logger = logging.getLogger(__name__)

class CalvinEvaluationEnv:
    """
    Unified evaluation interface for the Calvin environment.
    """
    
    def __init__(self, 
                 dataset_path: str, 
                 show_gui: bool = False,
                 num_sequences: int = 1000):
        """
        Initialize Calvin evaluation environment.
        
        Args:
            dataset_path: Calvin dataset path
            show_gui: Whether to show GUI
            num_sequences: Number of evaluation sequences
        """
        self.dataset_path = Path(dataset_path)
        self.show_gui = show_gui
        self.num_sequences = num_sequences
        self.env = None
        self.current_sequence_idx = 0
        self.current_subtask_idx = 0
        self.current_step = 0
        self.max_episode_length = 360
        
        # Load configuration
        self._load_config()
        
        # Create environment
        self._create_env()
        
        # Load evaluation sequences
        self.eval_sequences = get_sequences(num_sequences)
        self.total_sequences = len(self.eval_sequences)
        
        # Evaluation results storage
        self.results = []
        self.current_sequence_success = 0
        
        logger.info(f"Calvin evaluation environment initialized with {self.total_sequences} evaluation sequences.")
    
    def _load_config(self):
        """Load Calvin configuration files"""
        # conf_dir = Path(__file__).absolute().parents[2] / "conf"
        conf_dir = Path("/data/sea_disk0/zhangxx/calvin/calvin_models")/"conf"
        self.task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
        self.task_oracle = hydra.utils.instantiate(self.task_cfg)
        self.val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")
    
    def _create_env(self):
        """Create Calvin environment"""
        val_folder = self.dataset_path / "validation"
        self.env = get_env(val_folder, show_gui=self.show_gui)
        logger.info("Calvin environment created successfully.")
    
    def reset_sequence(self) -> Tuple[Dict[str, Any], str, bool]:
        """
        Reset to the next evaluation sequence.
        
        Returns:
            observation: Environment observation
            instruction: Current language instruction
            done: Whether all sequences are completed
        """
        if self.current_sequence_idx >= self.total_sequences:
            logger.info("All evaluation sequences completed.")
            return None, None, True
        
        # Save the results of the previous sequence
        if self.current_sequence_idx > 0:
            self.results.append(self.current_sequence_success)
        
        # Get current sequence
        initial_state, eval_sequence = self.eval_sequences[self.current_sequence_idx]
        self.current_eval_sequence = eval_sequence
        self.current_subtask_idx = 0
        self.current_sequence_success = 0
        self.current_step = 0
        
        # Reset environment with the initial state
        robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
        self.env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
        
        # Get the instruction for the first subtask
        current_subtask = self.current_eval_sequence[self.current_subtask_idx]
        instruction = self.val_annotations[current_subtask][0]
        
        # Get observation
        observation = self.env.get_obs()
        
        # Get info of the task before starting
        self.start_info = self.env.get_info()

        logger.info(f"Sequence {self.current_sequence_idx + 1}/{self.total_sequences}: {' -> '.join(self.current_eval_sequence)}")
        logger.info(f"Current subtask: {current_subtask}")
        
        return observation, instruction, False
    
    def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], str, bool, bool, Dict[str, Any]]:
        """
        Execute an action in the environment and return the new observation, instruction, and task status.
        
        Args:
            action: Predicted action to be executed in the environment
            
        Returns:
            observation: New environment observation after executing the action
            instruction: Current lang instruction (Maybe updated to next subtask)
            subtask_done: Whether the current subtask is completed 
            sequence_done: Whether the current sequence is completed
            info: Additional information including sequence index, subtask index, step count, etc.
        """
        if self.current_sequence_idx >= self.total_sequences:
            raise RuntimeError("All sequences have been completed. Please call reset_sequence() to start over.")
        
        # Do action
        obs, _, _, current_info = self.env.step(action)
        self.current_step += 1
        
        # Check if the current subtask is done
        current_subtask = self.current_eval_sequence[self.current_subtask_idx]
        current_task_info = self.task_oracle.get_task_info_for_set(
            self.start_info, current_info, {current_subtask}
        )
        
        subtask_done = len(current_task_info) > 0
        sequence_done = False
        instruction = self.val_annotations[current_subtask][0]
        
        info = {
            'sequence_idx': self.current_sequence_idx,
            'subtask_idx': self.current_subtask_idx,
            'step': self.current_step,
            'subtask': current_subtask,
            'max_steps_reached': self.current_step >= self.max_episode_length
        }
        
        if subtask_done:
            # Current subtask completed successfully
            self.current_sequence_success += 1
            self.current_subtask_idx += 1
            self.current_step = 0  # Reset step counter
            
            logger.info(f"Subtask completed: {current_subtask}")
            
            # Check if there are more subtasks.
            if self.current_subtask_idx >= len(self.current_eval_sequence):
                # Sequence done
                sequence_done = True
                self.current_sequence_idx += 1
                logger.info(f"Sequence completed! Successfully finished {self.current_sequence_success}/{len(self.current_eval_sequence)} subtasks.")
            else:
                # Go on to the next subtask
                next_subtask = self.current_eval_sequence[self.current_subtask_idx]
                instruction = self.val_annotations[next_subtask][0]
                self.start_info = current_info  # Update start info for the next subtask
                logger.info(f"Next subtask: {next_subtask}")
                
        elif self.current_step >= self.max_episode_length:
            # Max steps reached. Subtask failed.
            sequence_done = True
            self.current_sequence_idx += 1
            logger.info(f"Sequence failed! Successfully finished {self.current_sequence_success}/{len(self.current_eval_sequence)} subtasks.")
        
        return obs, instruction, subtask_done, sequence_done, info
    
    def get_current_instruction(self) -> str:
        """Get current language instruction"""
        if self.current_sequence_idx >= self.total_sequences:
            return ""
        current_subtask = self.current_eval_sequence[self.current_subtask_idx]
        return self.val_annotations[current_subtask][0]
    
    def get_evaluation_results(self) -> Dict[str, Any]:
        """
        Get evaluation results including success rates and statistics.
        
        Returns:
            Dict with evaluation statistics.
        """
        if len(self.results) == 0:
            return {"message": "No evaluation sequences completed yet."}
        
        # Calculate success rates for different lengths
        success_rates = []
        for length in range(1, 6):
            successes = sum(1 for result in self.results if result >= length)
            success_rates.append(successes / len(self.results))
        
        total_tasks = sum(self.results)
        total_possible = len(self.results) * 5  # Suppose each sequence has a maximum of 5 subtasks
        
        return {
            'total_sequences_evaluated': len(self.results),
            'success_rates': success_rates,
            'success_rates_percent': [rate * 100 for rate in success_rates],
            'average_success_length': sum(self.results) / len(self.results) if self.results else 0,
            'total_tasks_completed': total_tasks,
            'total_possible_tasks': total_possible,
            'overall_task_success_rate': total_tasks / total_possible if total_possible > 0 else 0
        }
    
    def render(self, mode: str = "rgb_array") -> np.ndarray:
        return self.env.render(mode=mode)
    
    def close(self):
        if self.env is not None:
            self.env.close()