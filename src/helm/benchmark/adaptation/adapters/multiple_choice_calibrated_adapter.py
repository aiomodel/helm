from dataclasses import replace
from typing import List

from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.scenarios.scenario import Instance, Input
from .multiple_choice_separate_adapter import MultipleChoiceSeparateAdapter


class MultipleChoiceCalibratedAdapter(MultipleChoiceSeparateAdapter):
    """
    Each answer choice sentence is scored independently, where the score is the sentence probability
    normalized by the unconditional sentence probability.
    For more details, refer to Section 2.4 of the GPT-3 paper (https://arxiv.org/pdf/2005.14165.pdf).
    """

    def generate_requests(self, eval_instance: Instance) -> List[RequestState]:
        request_states: List[RequestState] = []

        for reference_index, reference in enumerate(eval_instance.references):
            # original
            prompt = self.construct_prompt(
                self.train_instances,
                eval_instance,
                include_output=True,
                reference_index=reference_index,
            )
            request_states.append(self.construct_request_state(prompt, reference_index, eval_instance))

            # calibration
            # Compute the logprobs of the reference without train instances and the input question.
            new_eval_instance = replace(eval_instance, input=Input(text="Answer:"))
            prompt = self.construct_prompt(
                [],
                replace(eval_instance, input=Input(text="Answer:")),
                include_output=True,
                reference_index=reference_index,
            )
            request_states.append(
                self.construct_request_state(prompt, reference_index, eval_instance, request_mode="calibration", neval_instance=new_eval_instance)
            )

        return request_states
