from typing import List

from helm.benchmark.adaptation.prompt import Prompt
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.scenarios.scenario import Instance
from helm.common.request import Request
from .in_context_learning_adapter import InContextLearningAdapter


class MultipleChoiceSeparateAdapter(InContextLearningAdapter):
    """
    Each answer choice sentence is scored independently, where the score is
    the sentence probability normalized by the sentence length.
    """

    def generate_requests(self, eval_instance: Instance) -> List[RequestState]:
        request_states: List[RequestState] = []

        for reference_index, reference in enumerate(eval_instance.references):
            prompt = self.construct_prompt(
                self.train_instances,
                eval_instance,
                include_output=True,
                reference_index=reference_index,
            )
            request_states.append(self.construct_request_state(prompt, reference_index, eval_instance))

        return request_states

    def construct_request_state(
        self, prompt: Prompt, reference_index: int, eval_instance: Instance, request_mode: str = "original", neval_instance = None
    ) -> RequestState:
        request = Request(
            model=self.adapter_spec.model,
            prompt=prompt.text,
            num_completions=1,
            temperature=0,
            max_tokens=0,
            stop_sequences=[],
            echo_prompt=True,
            additional={"instance": eval_instance.input.text} if request_mode == "original" else {"instance": neval_instance.input.text},  # Input.text is the input;
        )
        return RequestState(
            instance=eval_instance,
            reference_index=reference_index,
            request_mode=request_mode,
            train_trial_index=self.train_trial_index,
            output_mapping=None,
            request=request,
            result=None,
            num_train_instances=prompt.num_train_instances,
            prompt_truncated=prompt.truncated,
        )
