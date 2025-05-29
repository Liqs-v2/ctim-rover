"""
The main driver.
"""

import ast
import json
import os
import re
import shutil
from argparse import ArgumentParser
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from itertools import chain
from os.path import abspath
from os.path import join as pjoin
from textwrap import indent

from loguru import logger

from pymilvus import MilvusClient
from app import globals, globals_mut, inference, log
from app import utils as apputils
from app.api.manage import ProjectApiManager
from app.model import common
from app.model.register import register_all_models
from app.post_process import (
    extract_organize_and_form_input,
    get_final_patch_path,
    organize_and_form_input,
    reextract_organize_and_form_inputs,
)
from app.raw_tasks import RawGithubTask, RawLocalTask, RawSweTask, RawTask
from app.task import Task


GENERAL_RULESET_PROMPT_TEMPLATE="""
Below follows a set of general, high-level insights extracted from successful trajectories in previous episodes, 
which you can use to plan you actions during the bug localization and improve your patch generation.
The insights are delimited by the <general_insights></general_insights> tags.
<general_insights>
{general_ruleset}
</general_insights>
"""

REPOSITORY_LEVEL_RULESET_PROMPT_TEMPLATE="""
In addition to these general insights, you are provided repository-level insights extracted from past successful trajectories 
from the {repository_name} repository, which your current task concerns itself with. Use these insights to improve your planning 
during bug localization and patch generation. These insights relate to understanding the project structure, data flow, 
repository-specific coding conventions, architectural and design patterns or common  
failure modes in the application domain of this software project. The insights are delimited by the <repository_level_insights></repository_level_insights> tags.
<repository_level_insights>
{repository_level_ruleset}
</repository_level_insights>
"""

SELF_REFLECTION_PROMPT_TEMPLATE="""Below follow self-reflections from previous attempts in which you failed to fix this bug. Consider these insights during your 
next attempt as they may help you avoid similar mistakes or leverage successful strategies. The reflections are 
delimited by <self_reflection-i></self_reflection-i> tags.
{self_reflections}
"""

def get_args(
        from_command_line_str: str = None, subparser_dest_attr_name: str = "command"
):
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(dest=subparser_dest_attr_name)

    swe_parser = subparsers.add_parser(
        "swe-bench", help="Run one or multiple swe-bench tasks"
    )
    set_swe_parser_args(swe_parser)

    github_parser = subparsers.add_parser(
        "github-issue", help="Run an online github issue"
    )
    set_github_parser_args(github_parser)

    local_parser = subparsers.add_parser("local-issue", help="Run a local issue.")
    set_local_parser_args(local_parser)

    extract_patches_parser = subparsers.add_parser(
        "extract-patches", help="Only extract patches from the raw results dir"
    )
    extract_patches_parser.add_argument("experiment-dir", type=str)

    re_extract_patches_parser = subparsers.add_parser(
        "re-extract-patches",
        help=(
            "same as extract-patches, except that individual dirs"
            " are moved out of their categories first"
        ),
    )
    re_extract_patches_parser.add_argument("experiment-dir", type=str)

    if not from_command_line_str:
        return parser.parse_args()
    return parser.parse_args(from_command_line_str.split())


def main(args, subparser_dest_attr_name: str = "command"):
    ## common options
    globals.output_dir = args.output_dir
    if globals.output_dir is not None:
        globals.output_dir = abspath(globals.output_dir)
    num_processes: int = int(args.num_processes)
    # set whether brief or verbose log
    print_stdout: bool = not args.no_print
    log.print_stdout = print_stdout

    common.set_model(args.model)
    # FIXME: make temperature part of the Model class
    common.MODEL_TEMP = args.model_temperature
    # acr related
    globals.conv_round_limit = args.conv_round_limit
    globals.enable_layered = args.enable_layered
    globals.enable_sbfl = args.enable_sbfl
    globals.enable_validation = args.enable_validation
    globals.enable_angelic = args.enable_angelic
    globals.enable_perfect_angelic = args.enable_perfect_angelic
    globals.only_save_sbfl_result = args.save_sbfl_result
    globals.disable_patch_generation = args.output_fix_locs
    globals.context_generation_limit = args.output_fix_limit
    globals.provide_exemplar = args.provide_exemplar
    globals.provide_general_cross_episode_knowledge = args.provide_general_cross_episode_knowledge
    globals.provide_repository_level_cross_episode_knowledge = args.provide_repository_level_cross_episode_knowledge
    globals.use_reflections = args.use_reflections

    subcommand = getattr(args, subparser_dest_attr_name)
    if subcommand == "swe-bench":
        tasks = make_swe_tasks(
            args.task, args.task_list_file, args.setup_map, args.tasks_map
        )

        groups = group_swe_tasks_by_env(tasks)
        run_task_groups(groups, num_processes, organize_output=True)
    elif subcommand == "github-issue":
        setup_dir = args.setup_dir
        if setup_dir is not None:
            setup_dir = abspath(setup_dir)

        task = RawGithubTask(
            args.task_id,
            args.clone_link,
            args.commit_hash,
            args.issue_link,
            setup_dir,
            args.use_comments,
        )
        groups = {"github": [task]}
        run_task_groups(groups, num_processes)
    elif subcommand == "local-issue":
        local_repo = args.local_repo
        if local_repo is not None:
            local_repo = abspath(local_repo)
        issue_file = args.issue_file
        if issue_file is not None:
            issue_file = abspath(issue_file)
        task = RawLocalTask(args.task_id, local_repo, issue_file)
        groups = {"local": [task]}
        run_task_groups(groups, num_processes)
    elif subcommand == "extract-patches":
        extract_organize_and_form_input(args.experiment_dir)
    elif subcommand == "re-extract-patches":
        reextract_organize_and_form_inputs(args.experiment_dir)


def set_swe_parser_args(parser: ArgumentParser) -> None:
    add_task_related_args(parser)

    parser.add_argument(
        "--setup-map",
        type=str,
        help="Path to json file that contains the setup information of the projects.",
    )
    parser.add_argument(
        "--tasks-map",
        type=str,
        help="Path to json file that contains the tasks information.",
    )
    parser.add_argument(
        "--task-list-file",
        type=str,
        help="Path to the file that contains all tasks ids to be run.",
    )
    parser.add_argument("--task", type=str, help="Task id to be run.")


def set_github_parser_args(parser: ArgumentParser) -> None:
    add_task_related_args(parser)
    parser.add_argument(
        "--task-id", type=str, help="Assign an id to the current fresh issue task."
    )
    parser.add_argument(
        "--clone-link", type=str, help="The link to the repository to clone."
    )
    parser.add_argument(
        "--commit-hash",
        type=str,
        help="The commit hash to checkout. If not specified, the latest commit on default branch will be used.",
    )
    parser.add_argument(
        "--use-comments",
        action="store_true",
        default=False,
        help="Include the comments of the issue.",
    )
    parser.add_argument("--issue-link", type=str, help="The link to the issue.")
    parser.add_argument(
        "--setup-dir",
        type=str,
        help="The directory where repositories should be cloned to.",
    )


def set_local_parser_args(parser: ArgumentParser) -> None:
    add_task_related_args(parser)
    parser.add_argument(
        "--task-id", type=str, help="Assign an id to the current local issue task."
    )
    parser.add_argument(
        "--local-repo", type=str, help="Path to a local copy of the target repo."
    )
    parser.add_argument("--issue-file", type=str, help="Path to a local issue file.")


def add_task_related_args(parser: ArgumentParser) -> None:
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Path to the directory that stores the run results.",
    )
    parser.add_argument(
        "--no-print",
        action="store_true",
        default=False,
        help="Do not print most messages to stdout.",
    )

    def model_parser(name: str):
        if not isinstance(name, str):
            raise TypeError(f"Invalid model name: {name}")
        if name in common.MODEL_HUB.keys():
            return name
        if name.startswith("litellm-generic-"):
            return name
        raise TypeError(f"Invalid model name: {name}")

    parser.add_argument(
        "--model",
        type=model_parser,
        default="gpt-3.5-turbo-0125",
        help="The model to use. Currently only OpenAI models are supported.",
    )
    parser.add_argument(
        "--model-temperature",
        type=float,
        default=0.0,
        help="The model temperature to use, for OpenAI models.",
    )
    parser.add_argument(
        "--conv-round-limit",
        type=int,
        default=15,
        help="Conversation round limit for the main agent.",
    )
    parser.add_argument(
        "--enable-layered",
        action="store_true",
        default=True,
        help="Enable layered code search.",
    )
    parser.add_argument(
        "--enable-sbfl", action="store_true", default=False, help="Enable SBFL."
    )
    parser.add_argument(
        "--enable-validation",
        action="store_true",
        default=False,
        help="Enable validation in our workflow.",
    )
    parser.add_argument(
        "--enable-angelic",
        action="store_true",
        default=False,
        help="(Experimental) Enable angelic debugging",
    )
    parser.add_argument(
        "--enable-perfect-angelic",
        action="store_true",
        default=False,
        help="(Experimental) Enable perfect angelic debugging; overrides --enable-angelic",
    )
    parser.add_argument(
        "--save-sbfl-result",
        action="store_true",
        default=False,
        help="Special mode to only save SBFL results for future runs.",
    )
    parser.add_argument(
        "--num-processes",
        type=str,
        default=1,
        help="Number of processes to run the tasks in parallel.",
    )
    parser.add_argument(
        "--output-fix-locs",
        action="store_true",
        required=False,
        default=False,
        help="Output fix locations to file and do not repair.",
    )
    parser.add_argument(
        "--output-fix-limit",
        type=int,
        required=False,
        default=10,
        help="Limit output of content retrieval rounds",
    )
    parser.add_argument(
        "--provide-exemplar",
        action="store_true",
        default=False,
        help="Provide an exemplar of the most task similar, successful run from SWE-Bench Lite whenever possible."
    )
    parser.add_argument(
        "--provide-general-cross-episode-knowledge",
        action="store_true",
        default=False,
        help="Provide a list of general, repository-agnostic insights distilled from SWE-Bench Lite trajectories to enhance problem solving."
    )
    parser.add_argument(
        "--provide-repository-level-cross-episode-knowledge",
        action="store_true",
        default=False,
        help="Provide a list of repository-specific insights distilled from SWE-Bench Lite trajectories to enhance problem solving."
    )
    parser.add_argument(
        "--use-reflections",
        action="store_true",
        default=False,
        help="Incorporate self-reflections from previous debugging attempts to guide current attempt."
    )


def make_swe_tasks(
        task_id: str | None,
        task_list_file: str | None,
        setup_map_file: str,
        tasks_map_file: str,
) -> list[RawSweTask]:
    if task_id is not None and task_list_file is not None:
        raise ValueError("Cannot specify both task and task-list.")

    all_task_ids = []
    if task_list_file is not None:
        all_task_ids = parse_task_list_file(task_list_file)
    if task_id is not None:
        all_task_ids = [task_id]
    if len(all_task_ids) == 0:
        raise ValueError("No task ids to run.")

    with open(setup_map_file) as f:
        setup_map = json.load(f)
    with open(tasks_map_file) as f:
        tasks_map = json.load(f)

    # Check if all task ids are in the setup and tasks map
    # This allows failing safely if some tasks are not set up properly
    missing_task_ids = [
        x for x in all_task_ids if not (x in setup_map and x in tasks_map)
    ]
    if missing_task_ids:
        # Log the tasks that are not in the setup or tasks map
        for task_id in sorted(missing_task_ids):
            log.print_with_time(
                f"Skipping task {task_id} which was not found in setup or tasks map."
            )
        # And drop them from the list of all task ids
        all_task_ids = filter(lambda x: x not in missing_task_ids, all_task_ids)

    all_task_ids = sorted(all_task_ids)

    # for each task in the list to run, create a Task instance
    all_tasks = []
    for task_id in all_task_ids:
        setup_info = setup_map[task_id]
        task_info = tasks_map[task_id]
        task = RawSweTask(task_id, setup_info, task_info)
        all_tasks.append(task)
    return all_tasks


def parse_task_list_file(task_list_file: str) -> list[str]:
    """
    Parse the task list file.
    The file should contain one task/instance id per line, without other characters.
    """
    with open(task_list_file) as f:
        task_ids = f.readlines()
    return [x.strip() for x in task_ids]


def group_swe_tasks_by_env(tasks: list[RawSweTask]) -> dict[str, list[RawSweTask]]:
    groups = {}
    for task in tasks:
        key = task.setup_info["env_name"]
        if key not in groups:
            groups[key] = []
        groups[key].append(task)
    return groups


def run_task_groups(
        task_groups: Mapping[str, Sequence[RawTask]],
        num_processes: int,
        organize_output: bool = False
):
    """
    Main entry for running tasks.
    """
    all_tasks = list(chain.from_iterable(task_groups.values()))
    num_tasks = len(all_tasks)

    globals_mut.init_total_num_tasks(num_tasks)

    # print some info about task
    log.print_with_time(f"Total number of tasks: {num_tasks}")
    log.print_with_time(f"Total number of processes: {num_processes}")
    log.print_with_time(f"Task group info: (number of groups: {len(task_groups)})")
    for key, tasks in task_groups.items():
        log.print_with_time(f"\t{key}: {len(tasks)} tasks")

    if globals.provide_exemplar:
        log.print_with_time("Running with in-context learning. Providing exemplars of most task similar, " +
                            "successful runs from SWE-Bench Lite whenever possible.")

        client = MilvusClient('data/task_embeddings.db')
        client.load_collection('swe_bench_lite')
        client.load_collection('swe_bench_verified')

        # Get query samples for all tasks. We need their problem_statement embeddings to find the most task similar trajectories
        # Needed for assosiating the task_id from SWE-Bench Verified with the most task similar sample from SWE-Bench Lite by zipping
        task_ids = [task.task_id for task in all_tasks]
        instance_ids = ','.join([f'"{task_id}"' for task_id in task_ids])
        filter_string = f'instance_id in [{instance_ids}]'

        query_samples = client.query(
            collection_name='swe_bench_verified',
            limit=num_tasks,
            filter=filter_string
        )

        query_vectors = [sample['vector'] for sample in query_samples]
        
        solved_swe_bench_verified_instances = json.load(open('data/all_successful_patch_trajectories.json', encoding='utf-8'))
        solved_swe_bench_verified_instances = [s['instance_id'] for s in solved_swe_bench_verified_instances if s['source'] == 'swe_bench_verified']
        
        # Construct proper filter string for Milvus with solved instances
        solved_instances_quoted = ','.join([f'"{instance_id}"' for instance_id in solved_swe_bench_verified_instances])
        filter_string_verified = f'instance_id in [{solved_instances_quoted}]'

        task_similarity_results_lite = client.search(collection_name='swe_bench_lite', data=query_vectors, limit=1)
        task_similarity_results_verified = client.search(collection_name='swe_bench_verified', data=query_vectors, limit=1, 
                                                        filter=filter_string_verified)
        
        # Throughout this entire section the order of samples in the lists are semantically important
        # They to which task the sample is the most similar to
        task_similarity_results_lite = [sample[0] for sample in task_similarity_results_lite] 
        task_similarity_results_verified = [sample[0] for sample in task_similarity_results_verified] # Train and test split are disjoing, so we will never find the sample itself in the results
        
        swe_bench_verified_ids_to_fetch = []
        swe_bench_lite_ids_to_fetch = []
        most_task_similar_tasks = []
        tasks = []
        for verified_sample, lite_sample in zip(task_similarity_results_verified, task_similarity_results_lite):
            if verified_sample['distance'] > lite_sample['distance']:
                swe_bench_verified_ids_to_fetch.append(verified_sample['id'])
                most_task_similar_tasks.append(verified_sample)
                tasks.append(verified_sample)
            else:
                swe_bench_lite_ids_to_fetch.append(lite_sample['id'])
                most_task_similar_tasks.append(lite_sample)
                tasks.append(lite_sample)
                
        most_task_similar_samples = client.get(collection_name='swe_bench_lite', ids=swe_bench_lite_ids_to_fetch) + \
            client.get(collection_name='swe_bench_verified', ids=swe_bench_verified_ids_to_fetch)

        task_to_most_similar_task_mapping = dict(zip(task_ids, most_task_similar_tasks))
        # Replace the sample_ids in task_to_most_similar_task_mapping with the actual samples from most_task_similar_samples
        # This is just a lookup dict, through which we access samples directly via their id, so order doesnt matter here
        lookup_dict = {sample['id']: sample for sample in most_task_similar_samples} 
        
        # And here the iteration is fed by an iterable that respects the order of tasks again
        task_to_most_similar_task_mapping = {task_id: {'sample': lookup_dict[most_task_similar_task['id']], 
                                                       'source': 'lite' if most_task_similar_task['id'] in swe_bench_lite_ids_to_fetch else 'verified',
                                                       'similarity': most_task_similar_task['distance']} for \
                                             task_id, most_task_similar_task in
                                             task_to_most_similar_task_mapping.items()}
        log.print_with_time("Successfully retrieved exemplars for all tasks.")
    else:
        task_to_most_similar_task_mapping = None

    if globals.provide_general_cross_episode_knowledge or globals.provide_repository_level_cross_episode_knowledge:
        path_to_ruleset = 'data/ruleset.json'
        ruleset = None
        if os.path.exists(path_to_ruleset):
            with open(path_to_ruleset, 'r') as f:
                ruleset = json.load(f)

        path_to_repo_ruleset = 'data/repo_ruleset.json'
        repo_level_ruleset = None
        if os.path.exists(path_to_repo_ruleset):
            with open(path_to_repo_ruleset, 'r') as f:
                repo_level_ruleset = json.load(f)

        rulesets = {'general': ruleset, 'repository_level': repo_level_ruleset}
    else:
        rulesets = None


    # single process mode
    if num_processes == 1:
        log.print_with_time("Running in single process mode.")
        run_tasks_serial(all_tasks, task_to_most_similar_task_mapping, rulesets)
        log.print_with_time("Finished all tasks sequentially.")
    else:
        run_task_groups_parallel(task_groups, num_processes, task_to_most_similar_task_mapping, rulesets)

    if globals.only_save_sbfl_result:
        log.print_with_time("Only saving SBFL results. Exiting.")
        return

    if organize_output:
        # post-process completed experiments to get input file to SWE-bench
        log.print_with_time("Post-processing completed experiment results.")
        swe_input_file = organize_and_form_input(globals.output_dir)
        log.print_with_time(f"SWE-Bench input file created: {swe_input_file}")


def run_tasks_serial(tasks: list[RawTask], task_to_most_similar_task_mapping=None, rulesets=None) -> None:
    swe_bench_verified_trajectories = json.load(open('data/all_successful_patch_trajectories.json', encoding='utf-8'))
    
    for task in tasks:
        most_similar_task = task_to_most_similar_task_mapping[
            task.task_id] if task_to_most_similar_task_mapping is not None else None
        
        if most_similar_task and most_similar_task['similarity'] > 0.9:
            most_similar_task_trajectory = [s for s in swe_bench_verified_trajectories if s['instance_id'] == most_similar_task['sample']['instance_id']][0]['patch_generation_trajectory'] \
                if most_similar_task['source'] == 'verified' else ast.literal_eval(most_similar_task['sample']['patch_generation_trajectory'])
        else:
            most_similar_task_trajectory = None

        if (globals.provide_general_cross_episode_knowledge or globals.provide_repository_level_cross_episode_knowledge) and rulesets is not None:
            repository = task.task_info['repo'].split('/')[0]
            ruleset_prompt = _build_ruleset_prompt_for(repository, rulesets)
        else:
            ruleset_prompt = None

        run_task_in_subprocess(task, most_similar_task_trajectory, ruleset_prompt)


def _build_ruleset_prompt_for(task_repository, rulesets):
    ruleset_prompt = ""
    if globals.provide_general_cross_episode_knowledge:
        general_ruleset = [{k: v for k, v in rule.items() if k != 'importance'} for rule in rulesets['general']]
        ruleset_prompt = GENERAL_RULESET_PROMPT_TEMPLATE.format(
            general_ruleset=indent('\n'.join(f"Insight {rule['id']}: {rule['content']}" for rule in general_ruleset), '\t'))
    if globals.provide_repository_level_cross_episode_knowledge and task_repository in rulesets['repository_level']:
        repository_level_ruleset = rulesets['repository_level'][task_repository]
        repository_level_ruleset = [{k: v for k, v in rule.items() if k != 'importance'} for rule in
                                    repository_level_ruleset]

        repository_level_ruleset_prompt = REPOSITORY_LEVEL_RULESET_PROMPT_TEMPLATE.format(
            repository_level_ruleset=indent('\n'.join(f"Insight {rule['id']} of type {rule['knowledge_type']}: {rule['content']}"
                                             for rule in repository_level_ruleset), '\t'),
            repository_name=task_repository)
        ruleset_prompt += repository_level_ruleset_prompt
    return ruleset_prompt


def run_task_groups_parallel(
        task_groups: Mapping[str, Sequence[RawTask]], num_processes: int, task_to_most_similar_task_mapping=None, rulesets=None
):
    num_task_groups = len(task_groups)
    globals_mut.init_total_num_task_groups(num_task_groups)
    num_processes = min(num_processes, num_task_groups)

    task_group_ids_items = sorted(
        task_groups.items(), key=lambda x: len(x[1]), reverse=True
    )
    log.print_with_time(f"Sorted task groups: {[x[0] for x in task_group_ids_items]}")
    try:
        # Use ProcessPoolExecutor instead of multiprocessing.Pool,
        # to support nested sub-processing

        group_ids, group_tasks = zip(*task_group_ids_items)
        with ProcessPoolExecutor(num_processes) as executor:
            # Reading Note: Map will call fn with the current item in the iterables. This means they are automatically unpacked in the callee fn.
            #   But we do not know yet which task_id we are currently processing, thus we need to pass the entire mapping.
            executor.map(run_task_group, group_ids, group_tasks, [task_to_most_similar_task_mapping] * len(group_ids), [rulesets] * len(group_ids))
    finally:
        log.print_with_time("Finishing all tasks in the pool.")


def run_task_group(task_group_id: str, task_group_items: list[RawTask], task_to_most_similar_task_mapping=None, rulesets=None) -> None:
    """
    Run all tasks in a task group sequentially.
    Main entry to parallel processing.
    """
    log.print_with_time(
        f"Starting process for task group {task_group_id}. Number of tasks: {len(task_group_items)}."
    )
    
    swe_bench_verified_trajectories = json.load(open('data/all_successful_patch_trajectories.json', encoding='utf-8'))
    
    for task in task_group_items:
        # within a group, the runs are always sequential
        most_similar_task = task_to_most_similar_task_mapping[
            task.task_id] if task_to_most_similar_task_mapping is not None else None
        
        if most_similar_task and most_similar_task['similarity'] > 0.9:
            most_similar_task_trajectory = [s for s in swe_bench_verified_trajectories if s['instance_id'] == most_similar_task['sample']['instance_id']][0]['patch_generation_trajectory'] \
                if most_similar_task['source'] == 'verified' else ast.literal_eval(most_similar_task['sample']['patch_generation_trajectory'])
        else:
            most_similar_task_trajectory = None

        if (globals.provide_general_cross_episode_knowledge or globals.provide_repository_level_cross_episode_knowledge) and rulesets is not None:
            repository = task.task_info['repo'].split('/')[0]
            ruleset_prompt = _build_ruleset_prompt_for(repository, rulesets)
        else:
            ruleset_prompt = None

        run_task_in_subprocess(task, most_similar_task_trajectory, ruleset_prompt)
        log.print_with_time(globals_mut.incre_task_return_msg())

    log.print_with_time(
        f"{globals_mut.incre_task_group_return_msg()} Finished task group {task_group_id}."
    )


def run_task_in_subprocess(task: RawTask, most_similar_task=None, ruleset_prompt=None) -> None:
    with ProcessPoolExecutor(max_workers=1) as executor:
        executor.submit(run_raw_task, task, None, most_similar_task=most_similar_task, ruleset_prompt=ruleset_prompt)


def run_raw_task(
        task: RawTask, print_callback: Callable[[dict], None] | None = None, most_similar_task=None, ruleset_prompt=None
) -> bool:
    """
    High-level entry for running one task.

    Args:
        - task: The Task instance to run.

    Returns:
        Whether the task completed successfully.
    """
    task_id = task.task_id

    start_time_s = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    task_output_dir = pjoin(globals.output_dir, f"{task_id}_{start_time_s}")
    apputils.create_dir_if_not_exists(task_output_dir)

    task.dump_meta_data(task_output_dir)

    log.log_and_always_print(f"============= Running task {task_id} =============")

    run_ok = False

    # Load self-reflections if enabled
    reflection_prompt = None
    if globals.use_reflections and globals.output_dir:
        reflection_file = pjoin(globals.output_dir, "self_reflections.json")
        if os.path.exists(reflection_file):
            try:
                with open(reflection_file, 'r') as f:
                    reflections = json.load(f)
                
                if task_id in reflections and reflections[task_id]:
                    self_reflections = ''
                    for index, reflection in enumerate(reflections[task_id]):
                        self_reflections += f'<self_reflection-{index}>\n{reflection}\n</self_reflection-{index}>\n'
                    reflection_prompt = SELF_REFLECTION_PROMPT_TEMPLATE.format(
                        self_reflections=self_reflections
                    )
                    log.print_with_time(f"Loaded self-reflection for task {task_id}")
            except Exception as e:
                log.print_with_time(f"Failed to load self-reflections: {e}")

    try:
        # Combine prompts if multiple are provided
        combined_prompt = ""
        if ruleset_prompt:
            combined_prompt += ruleset_prompt
        if reflection_prompt:
            combined_prompt += "\n" + reflection_prompt
            
        final_prompt = combined_prompt if combined_prompt else None
            
        run_ok = do_inference(task.to_task(), task_output_dir, print_callback, 
                             most_similar_task=most_similar_task, 
                             combined_prompt=final_prompt)

        if run_ok:
            run_status_message = f"Task {task_id} completed successfully."
        else:
            run_status_message = f"Task {task_id} failed without exception."
    except Exception as e:
        logger.exception(e)
        run_status_message = f"Task {task_id} failed with exception: {e}."

    log.log_and_always_print(run_status_message)

    if globals.disable_patch_generation:
        log.log_and_always_print(
            f"Patch generation is disabled. Please find fix locations at: {task_output_dir}/fix_locations.json"
        )
    else:
        output_patch_path = pjoin(task_output_dir, "final_patch.diff")
        final_patch_path = get_final_patch_path(task_output_dir)
        if final_patch_path is not None:
            # cppy the final patch to the fixed path
            shutil.copy2(final_patch_path, output_patch_path)

            log.log_and_always_print(
                f"Please find the generated patch at: {output_patch_path}"
            )

            if isinstance(task, RawSweTask):
                log.log_and_always_print(
                    "[SWE-bench mode] Note that the patch may be move to other paths in SWE-bench mode. "
                    "Please check the SWE-bench input file containing generated patches for all tasks."
                )
        else:
            log.log_and_always_print(
                "No patch generated. You can try running ACR again."
            )

    return run_ok


def do_inference(
        python_task: Task,
        task_output_dir: str,
        print_callback: Callable[[dict], None] | None = None,
        most_similar_task=None,
        combined_prompt=None
) -> bool:
    apputils.create_dir_if_not_exists(task_output_dir)

    logger.add(
        pjoin(task_output_dir, "info.log"),
        level="DEBUG",
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level>"
            " | <level>{message}</level>"
        ),
    )

    start_time = datetime.now()

    api_manager = ProjectApiManager(python_task, task_output_dir)

    try:
        if globals.only_save_sbfl_result:
            _, _, run_ok = api_manager.fault_localization()
        else:
            run_ok = inference.run_one_task(
                api_manager.output_dir,
                api_manager,
                python_task.get_issue_statement(),
                print_callback,
                most_similar_task=most_similar_task,
                combined_prompt=combined_prompt
            )

            api_manager.dump_tool_call_sequence_to_file()
            api_manager.dump_tool_call_layers_to_file()

            end_time = datetime.now()

            dump_cost(start_time, end_time, task_output_dir, python_task.project_path)
    finally:
        python_task.reset_project()

    return run_ok


def dump_cost(
        start_time: datetime, end_time: datetime, task_output_dir: str, project_path: str
):
    with apputils.cd(project_path):
        commit_hash = apputils.get_current_commit_hash()
    model_stats = common.SELECTED_MODEL.get_overall_exec_stats()
    stats = {
        "commit": commit_hash,
        "start_epoch": start_time.timestamp(),
        "end_epoch": end_time.timestamp(),
        "elapsed_seconds": (end_time - start_time).total_seconds(),
    }
    stats.update(model_stats)

    with open(pjoin(task_output_dir, "cost.json"), "w") as f:
        json.dump(stats, f, indent=4)


if __name__ == "__main__":
    logger.remove()
    register_all_models()
    args = get_args()
    main(args)
