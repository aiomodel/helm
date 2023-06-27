import dacite
import json
import os
import traceback
import typing
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List

from tqdm import tqdm

from helm.common.general import ensure_directory_exists, write, asdict_without_nones
from helm.common.hierarchical_logger import hlog, htrack_block
from helm.common.cache import cache_stats
from .augmentations.data_augmenter import DataAugmenterSpec
from .scenarios.scenario import Scenario, ScenarioSpec, create_scenario, Instance, with_instance_ids
from .adaptation.adapters.adapter import Adapter
from .adaptation.adapters.adapter_factory import AdapterFactory
from .adaptation.scenario_state import ScenarioState
from .adaptation.adapter_spec import AdapterSpec
from .data_preprocessor import DataPreprocessor
from .executor import ExecutionSpec, Executor
from .metrics.dry_run_metrics import DryRunMetric
from .metrics.metric_name import MetricName
from .metrics.metric_service import MetricService
from .metrics.metric import Metric, MetricSpec, MetricResult, PerInstanceStats, create_metric, Stat
from .window_services.tokenizer_service import TokenizerService


LATEST_SYMLINK: str = "latest"


class RunnerError(Exception):
    """Error that happens in the Runner."""

    pass


@dataclass(frozen=True)
class RunSpec:
    """
    Specifies how to do a single run, which gets a scenario, adapts it, and
    computes a list of stats based on the defined metrics.
    """

    # Unique identifier of the RunSpec
    name: str

    # Which scenario
    scenario_spec: ScenarioSpec

    # Specifies how to adapt an instance into a set of requests
    adapter_spec: AdapterSpec

    # What to evaluate on
    metric_specs: List[MetricSpec]

    # Data augmenter. The default `DataAugmenterSpec` does nothing.
    data_augmenter_spec: DataAugmenterSpec = DataAugmenterSpec()

    # Groups that this run spec belongs to (for aggregation)
    groups: List[str] = field(default_factory=list)

    def __post_init__(self):
        """
        `self.name` is used as the name of the output folder for the `RunSpec`.
        Clean up `self.name` by replacing any "/"'s with "_".
        """
        # TODO: Don't mutate name! clean this up before passing it into the constructor here
        object.__setattr__(self, "name", self.name.replace(os.path.sep, "_"))


class Runner:
    """
    The main entry point for running the entire benchmark.  Mostly just
    dispatches to other classes.
    """

    def __init__(
        self,
        execution_spec: ExecutionSpec,
        output_path: str,
        suite: str,
        skip_instances: bool,
        cache_instances: bool,
        cache_instances_only: bool,
        skip_completed_runs: bool,
        exit_on_error: bool,
    ):
        self.executor = Executor(execution_spec)
        self.dry_run: bool = execution_spec.dry_run
        self.tokenizer_service = TokenizerService(self.executor.service, execution_spec.auth)
        self.metric_service = MetricService(self.executor.service, execution_spec.auth)
        self.skip_instances: bool = skip_instances
        self.cache_instances: bool = cache_instances
        self.cache_instances_only: bool = cache_instances_only
        self.skip_completed_runs: bool = skip_completed_runs
        self.exit_on_error: bool = exit_on_error

        ensure_directory_exists(output_path)
        # Decide where to save the raw data (e.g., "output/scenarios/mmlu").
        self.scenarios_path: str = os.path.join(output_path, "scenarios")
        ensure_directory_exists(self.scenarios_path)
        # Decide where to save input instances
        self.instances_path: str = os.path.join(output_path, "scenario_instances")
        ensure_directory_exists(self.instances_path)

        # Output the results under a folder with the name of the suite
        self.runs_path: str = os.path.join(output_path, "runs", suite)

        # The path where to cache files needs to compute metrics, e.g., human evaluation results
        self.eval_cache_path: str = os.path.join(self.runs_path, "eval_cache")
        ensure_directory_exists(self.eval_cache_path)

    def _is_run_completed(self, run_spec: RunSpec):
        """Return whether the run was previously completed.

        A run is completed if all of the expected output files exist."""
        run_path: str = os.path.join(self.runs_path, run_spec.name)
        if not os.path.isdir(run_path):
            return False
        output_paths = [
            os.path.join(run_path, "run_spec.json"),
            os.path.join(run_path, "scenario.json"),
            os.path.join(run_path, "scenario_state.json"),
            os.path.join(run_path, "stats.json"),
            os.path.join(run_path, "per_instance_stats.json"),
        ]
        for output_path in output_paths:
            if not os.path.exists(output_path):
                return False
        return True

    def run_all(self, run_specs: List[RunSpec]):
        failed_run_specs: List[RunSpec] = []

        for run_spec in tqdm(run_specs, disable=None):
            try:
                with htrack_block(f"Running {run_spec.name}"):
                    self.run_one(run_spec)
            except Exception as e:
                if self.exit_on_error:
                    raise e
                else:
                    hlog(f"Error when running {run_spec.name}:\n{traceback.format_exc()}")
                    failed_run_specs.append(run_spec)
        if not self.exit_on_error and failed_run_specs:
            failed_runs_str = ", ".join([f'"{run_spec.name}"' for run_spec in failed_run_specs])
            raise RunnerError(f"Failed runs: [{failed_runs_str}]")

    def run_one(self, run_spec: RunSpec):
        # Load the scenario
        scenario: Scenario = create_scenario(run_spec.scenario_spec)

        # This `output_path` will be used when `Adapter` calls `Scenario.get_instances`.
        scenario.output_path = os.path.join(self.scenarios_path, scenario.name)
        ensure_directory_exists(scenario.output_path)

        # This 'output_path' will be used when the model's input instances are saved.
        args_str = ",".join([f"{k}={v}" for k, v in sorted(run_spec.scenario_spec.args.items())])
        scenario_name_with_args = f"{scenario.name}:{args_str}" if args_str else f"{scenario.name}"
        input_instances_output_path = os.path.join(self.instances_path, scenario_name_with_args)
        input_instances_file_path = os.path.join(input_instances_output_path, "input_instances.json")

        run_path: str = os.path.join(self.runs_path, run_spec.name)
        ensure_directory_exists(run_path)

        if self.skip_completed_runs and self._is_run_completed(run_spec):
            # If scenario_state.json exists, assume that all other output files exist
            # because scenario_state.json is the last output file to be written.
            hlog(f"Skipping run {run_spec.name} because run is completed and all output files exist.")
            return

        # Fetch and initialize the Adapter based on the `AdapterSpec`.
        adapter: Adapter = AdapterFactory.get_adapter(run_spec.adapter_spec, self.tokenizer_service)

        instances: List[Instance]
        if self.skip_instances:
            instances = []
        else:
            if self.cache_instances and os.path.exists(input_instances_file_path):
                with open(input_instances_file_path) as f:
                    json_instances: List[Dict[str, Any]] = json.load(f)
                instances = [dacite.from_dict(Instance, instance) for instance in json_instances]
            else:
                # Create the instances of the scenario
                with htrack_block("scenario.get_instances"):
                    instances = scenario.get_instances()  # full instances (include train/val)
                    # Eg from hellaswag. Instance(input=Input(text='Capoeira: A group of men are holding maracas in the their hands and playing to some reggae music. One of the men'), references=[Reference(output=Output(text='gives the other man a hard time.'), tags=[]), Reference(output=Output(text='is out of his gear and showcases a trimmed razor.'), tags=[]), Reference(output=Output(text='kicks the other to the ground.'), tags=[]), Reference(output=Output(text='begins singing in the microphone.'), tags=['correct'])], split='train', sub_split=None, id=None, perturbation=None, contrast_inputs=None, contrast_references=None)
        if self.cache_instances and not os.path.exists(input_instances_file_path):
            # Save instances to file
            ensure_directory_exists(input_instances_output_path)
            write(
                os.path.join(input_instances_file_path),
                json.dumps([asdict_without_nones(instance) for instance in instances], indent=2),
            )
        if self.cache_instances_only:
            return  # Exit after saving the instances.

        # Give each instance a unique ID
        instances = with_instance_ids(instances)

        # Get the instances necessary for this run. Only select our the train and eval split by max_train_instances and max_eval_instances
        instances = adapter.get_run_instances(instances) # for hellaswag: helm.benchmark.adaptation.adapters.multiple_choice_separate_adapter.MultipleChoiceSeparateAdapter

        # Data preprocessing
        instances = DataPreprocessor(run_spec.data_augmenter_spec).preprocess(
            instances, self.executor.execution_spec.parallelism
        ) # for hellaswag: helm.benchmark.data_preprocessor.DataPreprocessor; Not much change

        # Adapt (convert to requests)
        scenario_state: ScenarioState = adapter.adapt(instances, self.executor.execution_spec.parallelism) # for hellaswag; eval instances -> 4*eval requests; requests = instances.input + one of instances.references
        # scenario_state.request_states[2].request: Request(model='ours/custom_gpt2_7b', embedding=False, prompt="Personal Care and Style: [header] How to dye your hair with semi permanent hair dye [title] Find the color you want. [step] There are many popular brands and hundreds of different colors to choose from. Semi-permanent dyes can be found in a variety of places, ranging from grocery stores to specialized fashion shops, with the biggest selection at beauty supply stores.  Pick the color that's your favorite, matches your wardrobe best, and/or is most flattering for your eye color and skin tone. Semi-permanent dyes work on all hair colors, but show up brightest on light hair.", temperature=0, num_completions=1, top_k_per_token=1, max_tokens=0, stop_sequences=[], echo_prompt=True, top_p=1, presence_penalty=0, frequency_penalty=0, random=None, messages=None)

        # Execute (fill up results) [truly call the models in client and server: make_request and serve_request]
        scenario_state = self.executor.execute(scenario_state)
        # scenario_state.request_states[2].result: result=RequestResult(success=True, embedding=[], completions=[Sequence(text="Personal Care and Style: [header] How to dye your hair with semi permanent hair dye [title] Find the color you want. [step] There are many popular brands and hundreds of different colors to choose from. Semi-permanent dyes can be found in a variety of places, ranging from grocery stores to specialized fashion shops, with the biggest selection at beauty supply stores.  Pick the color that's your favorite, matches your wardrobe best, and/or is most flattering for your eye color and skin tone. Semi-permanent dyes work on all hair colors, but show up brightest on light hair. [", logprob=0.0, tokens=[Token(text='Personal', logprob=0.0, top_logprobs={}), Token(text=' Care', logprob=0.0, top_logprobs={}), Token(text=' and', logprob=0.0, top_logprobs={}), Token(text=' Style', logprob=0.0, top_logprobs={}), Token(text=':', logprob=0.0, top_logprobs={}), Token(text=' [', logprob=0.0, top_logprobs={}), Token(text='header', logprob=0.0, top_logprobs={}), Token(text=']', logprob=0.0, top_logprobs={}), Token(text=' How', logprob=0.0, top_logprobs={}), Token(text=' to', logprob=0.0, top_logprobs={}), Token(text=' dye', logprob=0.0, top_logprobs={}), Token(text=' your', logprob=0.0, top_logprobs={}), Token(text=' hair', logprob=0.0, top_logprobs={}), Token(text=' with', logprob=0.0, top_logprobs={}), Token(text=' semi', logprob=0.0, top_logprobs={}), Token(text=' permanent', logprob=0.0, top_logprobs={}), Token(text=' hair', logprob=0.0, top_logprobs={}), Token(text=' dye', logprob=0.0, top_logprobs={}), Token(text=' [', logprob=0.0, top_logprobs={}), Token(text='title', logprob=0.0, top_logprobs={}), Token(text=']', logprob=0.0, top_logprobs={}), Token(text=' Find', logprob=0.0, top_logprobs={}), Token(text=' the', logprob=0.0, top_logprobs={}), Token(text=' color', logprob=0.0, top_logprobs={}), Token(text=' you', logprob=0.0, top_logprobs={}), Token(text=' want', logprob=0.0, top_logprobs={}), Token(text='.', logprob=0.0, top_logprobs={}), Token(text=' [', logprob=0.0, top_logprobs={}), Token(text='step', logprob=0.0, top_logprobs={}), Token(text=']', logprob=0.0, top_logprobs={}), Token(text=' There', logprob=0.0, top_logprobs={}), Token(text=' are', logprob=0.0, top_logprobs={}), Token(text=' many', logprob=0.0, top_logprobs={}), Token(text=' popular', logprob=0.0, top_logprobs={}), Token(text=' brands', logprob=0.0, top_logprobs={}), Token(text=' and', logprob=0.0, top_logprobs={}), Token(text=' hundreds', logprob=0.0, top_logprobs={}), Token(text=' of', logprob=0.0, top_logprobs={}), Token(text=' different', logprob=0.0, top_logprobs={}), Token(text=' colors', logprob=0.0, top_logprobs={}), Token(text=' to', logprob=0.0, top_logprobs={}), Token(text=' choose', logprob=0.0, top_logprobs={}), Token(text=' from', logprob=0.0, top_logprobs={}), Token(text='.', logprob=0.0, top_logprobs={}), Token(text=' Semi', logprob=0.0, top_logprobs={}), Token(text='-per', logprob=0.0, top_logprobs={}), Token(text='manent', logprob=0.0, top_logprobs={}), Token(text=' d', logprob=0.0, top_logprobs={}), Token(text='yes', logprob=0.0, top_logprobs={}), Token(text=' can', logprob=0.0, top_logprobs={}), Token(text=' be', logprob=0.0, top_logprobs={}), Token(text=' found', logprob=0.0, top_logprobs={}), Token(text=' in', logprob=0.0, top_logprobs={}), Token(text=' a', logprob=0.0, top_logprobs={}), Token(text=' variety', logprob=0.0, top_logprobs={}), Token(text=' of', logprob=0.0, top_logprobs={}), Token(text=' places', logprob=0.0, top_logprobs={}), Token(text=',', logprob=0.0, top_logprobs={}), Token(text=' ranging', logprob=0.0, top_logprobs={}), Token(text=' from', logprob=0.0, top_logprobs={}), Token(text=' grocery', logprob=0.0, top_logprobs={}), Token(text=' stores', logprob=0.0, top_logprobs={}), Token(text=' to', logprob=0.0, top_logprobs={}), Token(text=' specialized', logprob=0.0, top_logprobs={}), Token(text=' fashion', logprob=0.0, top_logprobs={}), Token(text=' shops', logprob=0.0, top_logprobs={}), Token(text=',', logprob=0.0, top_logprobs={}), Token(text=' with', logprob=0.0, top_logprobs={}), Token(text=' the', logprob=0.0, top_logprobs={}), Token(text=' biggest', logprob=0.0, top_logprobs={}), Token(text=' selection', logprob=0.0, top_logprobs={}), Token(text=' at', logprob=0.0, top_logprobs={}), Token(text=' beauty', logprob=0.0, top_logprobs={}), Token(text=' supply', logprob=0.0, top_logprobs={}), Token(text=' stores', logprob=0.0, top_logprobs={}), Token(text='.', logprob=0.0, top_logprobs={}), Token(text=' ', logprob=0.0, top_logprobs={}), Token(text=' Pick', logprob=0.0, top_logprobs={}), Token(text=' the', logprob=0.0, top_logprobs={}), Token(text=' color', logprob=0.0, top_logprobs={}), Token(text=' that', logprob=0.0, top_logprobs={}), Token(text="'s", logprob=0.0, top_logprobs={}), Token(text=' your', logprob=0.0, top_logprobs={}), Token(text=' favorite', logprob=0.0, top_logprobs={}), Token(text=',', logprob=0.0, top_logprobs={}), Token(text=' matches', logprob=0.0, top_logprobs={}), Token(text=' your', logprob=0.0, top_logprobs={}), Token(text=' wardrobe', logprob=0.0, top_logprobs={}), Token(text=' best', logprob=0.0, top_logprobs={}), Token(text=',', logprob=0.0, top_logprobs={}), Token(text=' and', logprob=0.0, top_logprobs={}), Token(text='/or', logprob=0.0, top_logprobs={}), Token(text=' is', logprob=0.0, top_logprobs={}), Token(text=' most', logprob=0.0, top_logprobs={}), Token(text=' flattering', logprob=0.0, top_logprobs={}), Token(text=' for', logprob=0.0, top_logprobs={}), Token(text=' your', logprob=0.0, top_logprobs={}), Token(text=' eye', logprob=0.0, top_logprobs={}), Token(text=' color', logprob=0.0, top_logprobs={}), Token(text=' and', logprob=0.0, top_logprobs={}), Token(text=' skin', logprob=0.0, top_logprobs={}), Token(text=' tone', logprob=0.0, top_logprobs={}), Token(text='.', logprob=0.0, top_logprobs={}), Token(text=' Semi', logprob=0.0, top_logprobs={}), Token(text='-per', logprob=0.0, top_logprobs={}), Token(text='manent', logprob=0.0, top_logprobs={}), Token(text=' d', logprob=0.0, top_logprobs={}), Token(text='yes', logprob=0.0, top_logprobs={}), Token(text=' work', logprob=0.0, top_logprobs={}), Token(text=' on', logprob=0.0, top_logprobs={}), Token(text=' all', logprob=0.0, top_logprobs={}), Token(text=' hair', logprob=0.0, top_logprobs={}), Token(text=' colors', logprob=0.0, top_logprobs={}), Token(text=',', logprob=0.0, top_logprobs={}), Token(text=' but', logprob=0.0, top_logprobs={}), Token(text=' show', logprob=0.0, top_logprobs={}), Token(text=' up', logprob=0.0, top_logprobs={}), Token(text=' brightest', logprob=0.0, top_logprobs={}), Token(text=' on', logprob=0.0, top_logprobs={}), Token(text=' light', logprob=0.0, top_logprobs={}), Token(text=' hair', logprob=0.0, top_logprobs={}), Token(text='.', logprob=0.0, top_logprobs={}), Token(text=' [', logprob=0.0, top_logprobs={' [': 0.0})], finish_reason=None)], cached=True, request_time=1.4849658012390137, request_datetime=1687011026, error=None, error_flags=None, batch_size=None, batch_request_time=None), num_train_instances=0, prompt_truncated=False, num_conditioning_tokens=0)

        # Apply the metrics
        # When performing a dry run, only estimate the number of tokens instead
        # of calculating the metrics.
        metrics: List[Metric] = (
            [DryRunMetric()] if self.dry_run else [create_metric(metric_spec) for metric_spec in run_spec.metric_specs]
        ) # for hellaswag: [BasicMetric(exact_match,quasi_exact_match,prefix_exact_match,quasi_prefix_exact_match)]
        stats: List[Stat] = []
        per_instance_stats: List[PerInstanceStats] = []
        with htrack_block(f"{len(metrics)} metrics"):
            for metric in metrics:
                with htrack_block(metric):
                    metric_result: MetricResult = metric.evaluate(
                        scenario_state,
                        self.metric_service, # helm.benchmark.metrics.metric_service.MetricService
                        self.eval_cache_path,
                        self.executor.execution_spec.parallelism,
                    )
                    stats.extend(metric_result.aggregated_stats)
                    per_instance_stats.extend(metric_result.per_instance_stats)

        # Check that there aren't duplicate `Stat`s
        # Note: doesn't catch near misses.
        metric_counts: typing.Counter[MetricName] = Counter([stat.name for stat in stats])
        for metric_name, count in metric_counts.items():
            if count > 1:
                hlog(f"WARNING: duplicate metric name {metric_name}")

        # Print out the number of stats
        hlog(f"Generated {len(stats)} stats.")

        if self.skip_instances:
            hlog("skip_instances was True. Skipping writing results out.")
            return

        # Output benchmarking information and results to files
        write(os.path.join(run_path, "run_spec.json"), json.dumps(asdict_without_nones(run_spec), indent=2))

        # Write out scenario
        write(os.path.join(run_path, "scenario.json"), json.dumps(asdict_without_nones(scenario), indent=2))

        # Write scenario state
        for i in range(len(scenario_state.request_states)):
            object.__setattr__(scenario_state.request_states[i].request, 'additional', None)
        write(os.path.join(run_path, "scenario_state.json"), json.dumps(asdict_without_nones(scenario_state), indent=2))

        write(
            os.path.join(run_path, "stats.json"), json.dumps([asdict_without_nones(stat) for stat in stats], indent=2)
        )
        write(
            os.path.join(run_path, "per_instance_stats.json"),
            json.dumps(list(map(asdict_without_nones, per_instance_stats)), indent=2),
        )

        cache_stats.print_status()
