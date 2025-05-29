import ast
import math
import os
import re
from random import shuffle, randint, seed
import json
from textwrap import indent
import argparse
from math import sqrt
from copy import deepcopy

SYSTEM_PROMPT = """You are an advanced reasoning agent that can ADD, EDIT, UPVOTE or DOWNVOTE rules 
    from an existing rule set, which is constructed by reflecting on and critiquing past successful task trajectories."""

OPERATION_EXPLANATION_PROMPT_TEMPLATE = """Provide the operations as a list
    containing JSON objects of the following schema:
    {{
        "operation_type": {{"enum": ["ADD", "EDIT", "UPVOTE", "DOWNVOTE"]}},
        "rule_id": {{"type": "integer"}},
        "rule_content": {{"type": "string"}}
    }}
    The "operation_type" field specifies the type of operation to perform on the rule with the given "rule_id". The "rule_id" 
    must be an integer identifying a rule in the current ruleset{ruleset_indices_hint}.
    If you are adding or editing a rule, additionally provide the "rule_content" field with the new content of the rule.

    Here is an example of a valid response:
    {{"operations":
        [{{
            "operation_type": "ADD",
            "rule_content": <Extracted insight, knowledge, tip or rule>
        }},
        {{
            "operation_type": "DOWNVOTE",
            "rule_id": <Integer identifying an EXISTING rule that is contradictory to another rule, this sample or too similar to another rule>
        }},
        {{
            "operation_type": "EDIT",
            "rule_id": <Integer identifying an EXISTING rule>,
            "rule_content": <Extracted insight, knowledge, tip or rule to update and enhance the EXISTING rule with>
        }}]
    }}

    Do not mention the trajectories or their ids explicitly in your responses. Do not reference specific file, class, 
    function or variable names to ensure that your ruleset is general and transferable to other task instances and  
    repositories. You can use any of the valid operation types multiple times. Each existing rule can be modified
    only once. The following operations are valid:
    - UPVOTE an EXISTING rule if it is strongly relevant in your current context and trajectories. Valid fields: [operation_type, rule_id]
    - DOWNVOTE an EXISTING rule if the rule contradicts your current context and trajectories or is similar to or a duplicate 
    of another existing rule. Make use of this operation to achieve a concise ruleset that is relevant across repositories and task instances. 
    If you downvote a rule often enough it will be removed from the ruleset. Valid fields: [operation_type, rule_id]
    - EDIT an EXISTING rule if it is not general enough or could be enhanced given your current context by rewriting, adding or removing content. Valid fields: [operation_type, rule_id, rule_content]
    - ADD a NEW rule if you identified insights that are generally applicable and transferable to other task instances. Make sure that the new rule is distinct 
    from existing rules. Valid fields: [operation_type, rule_content]

    Key requirements:
    - The only operation that is valid on rules that do not yet exist is ADD. 
    - If you have reached the maximum ruleset size, you must not add any new rules. Instead, you must edit existing rules or upvote/downvote existing rules.
    - You may provide between 1 and 4 operations.
"""

OPERATION_EXPLANATION_REPOSITORY_LEVEL_PROMPT_TEMPLATE = """Provide the operations as a list
    containing JSON objects of the following schema:
    {{
        "operation_type": {{"enum": ["ADD", "EDIT", "UPVOTE", "DOWNVOTE"]}},
        "rule_id": {{"type": "integer"}},
        "rule_content": {{"type": "string"}},
        "knowledge_type": {{"enum": ["repository_structure", "architectural_pattern", "coding_convention", "error_pattern",
                                        "application_domain"]}}
    }}
    The "operation_type" field specifies the type of operation to perform on the rule with the given "rule_id". The "rule_id" 
    must be an integer identifying a rule in the current ruleset{ruleset_indices_hint}.
    If you are adding or editing a rule, additionally provide the "rule_content" field with the new content of the rule.
    If you are adding a rule, you must also specify the "knowledge_type" of the rule.

    Here is an example of a valid response:
    {{"operations": 
        [{{
            "operation_type": "ADD",
            "rule_content": <Extracted insight, knowledge, tip or rule>,
            "knowledge_type": "error_pattern"
        }},
        {{
            "operation_type": "ADD",
            "rule_content": <Knowledge about the application domain of the project and typical edge cases resulting from this.>
            "knowledge_type": "application_domain"
        }},
        {{
            "operation_type": "DOWNVOTE",
            "rule_id": <Integer identifying an EXISTING rule that is contradictory to another rule, this sample or too similar to another rule>
        }},
        {{
            "operation_type": "EDIT",
            "rule_id": <Integer identifying an EXISTING rule>,
            "rule_content": <Extracted insight, knowledge, tip or rule to update and enhance the EXISTING rule with>
        }}]
    }}

    Do not mention the trajectories or their ids explicitly in your responses. You may reference specific file, class, 
    function names, but keep in mind that the repository evolves over time and files, classes or functions may be renamed,
    removed or refactored. You can use any of the valid operation types multiple times. Each existing rule can be modified
    only once. The following operations are valid:
    - UPVOTE an EXISTING rule if it is strongly relevant in your current context and trajectories. Valid fields: [operation_type, rule_id]
    - DOWNVOTE an EXISTING rule if the rule contradicts your current context and trajectories or it is similar to or a duplicate 
    of another existing rule (including general purpose rules). Make use of this operation to achieve a concise ruleset that is relevant across repositories and task instances. 
    If you downvote a rule often enough it will be removed from the ruleset. Valid fields: [operation_type, rule_id]
    - EDIT an EXISTING rule if it is not general enough or could be enhanced given your current context by rewriting, adding or removing content. Valid fields: [operation_type, rule_id, rule_content]
    - ADD a NEW rule if you identified insights that are generally applicable and potentially beneficial to other task instances in the same repository. Make sure that the new rule is unique.
     Valid fields: [operation_type, rule_content, knowledge_type]

    Key requirements:
    - The only operation that is valid on rules that do not yet exist is ADD. 
    - If you have reached the maximum ruleset size, you must not add any new rules. Instead, you must edit existing rules or upvote/downvote existing rules.
    - You may provide between 1 and 4 operations.
"""

DISTILLATION_FROM_SUCCESSFUL_TRAJECTORIES_PROMPT_TEMPLATE = """
    You are given a set of successful task trajectories that relate to fixing bugs in open-source code repositories. 
    During these trajectories you correctly identified the location of the buggy code, wrote
    a patch which fixed the bug in the code and passed all test cases, meaning you also didn't in introduce any new bugs.

    Below follow the past successful task trajectories. The set of trajectories is delimited by the <PAST_SUCCESSFUL_TRAJECTORIES> and </PAST_SUCCESSFUL_TRAJECTORIES> tags. 
    Each trajectory is wrapped by the <TRAJECTORY-i> and </TRAJECTORY-i> tags, where i identifies the i-th trajectory in the set below:
    <PAST_SUCCESSFUL_TRAJECTORIES>
{past_successful_trajectories}
    </PAST_SUCCESSFUL_TRAJECTORIES>

    Next, follow a set of rules that you have extracted so far. The ruleset is limited to {ruleset_cap} rules. Any rules beyond {ruleset_cap} rules will be ignored:
{current_ruleset}
{remaining_slots_information}

    By examining the successful trajectories, and the existing rules above you should update the existing ruleset by adding, editing, upvoting or downvoting rules. 
    The resulting ruleset must consist of high-level knowledge, insights or tips that are generally applicable, covering the following aspects:
    1. Reasoning and planning strategies that serve as guiding signals for future task attempts, especially with respect to identifying the locations of buggy code effectively.
    2. Coding practices, patterns, and idioms that are generally applicable to writing high-quality, staff senior level code, to fix bugs.
    3. Common pitfalls and error patterns in software engineering that are relevant to identifying and fixing buggy code.
    
    Key requirements for rules:
    - DO NOT suggest testing the implementation. The agent using your ruleset is UNABLE to test its implementation. It must generate a correct patch on the first attempt.
    - Generated rules must be concise (less than 80 words) and should be focused on a single, specific aspect or insight.
    - Generated rules must be unique with respect to other, existing rules and contribute a new, unique piece of information, knowledge or perspective.

    This ruleset should serve as the basis for guiding future task attempts in locating and fixing bugs to a successful completion and
    empower the agent to improve its planning, reasoning, coding skills, bug localization skills.
    
    """

DISTILLATION_FROM_SUCCESS_FAILURE_PAIRS_PROMPT_TEMPLATE = """
    Below you will find multiple past attempts at fixing a bug in an open-source code repository. The first few trajectories show failed attempts, the last trajectory shows a successful bug fix.
    All attempts are related to fixing the same bug in the same codebase.
    Compare and contrast the successful and failed attempts to understand why the initial attempts failed and 
    which change in the reasoning, planning, coding or bug localization strategy could have led to a correct patch generation in the first attempt.
    Consider the self-reflections that took place between the failed attempts to understand which changes were made in the reasoning, planning, coding or bug localization strategy that
    led to the bug being fixed in the last trajectory.

    Below follow the task attempts denoted by <FAILED_TASK_ATTEMPT-i> and </FAILED_TASK_ATTEMPT-i> tags where i identifies the i-th failed attempt and the 
    successful task attempt is denoted by the <SUCCESSFUL_TASK_ATTEMPT> and </SUCCESSFUL_TASK_ATTEMPT> tags. Only failed task attempts contain a self-reflection:
{success_failure_trajectory}

    Next, follow a set of rules that you have extracted so far. The ruleset is limited to {ruleset_cap} rules. Any rules beyond {ruleset_cap} rules will be ignored:
{current_ruleset}

{remaining_slots_information}

    By examining and comparing the successful and failed attempts, 
    and the existing rules above you should update the existing ruleset by adding, editing, upvoting or downvoting rules. 
    The resulting ruleset must consist of high-level knowledge, insights or tips that are generally applicable, covering the following aspects:
    1. Reasoning and planning strategies that serve as guiding signals for future task attempts, especially with respect to entifying the locations of buggy code effectively.
    2. Coding practices, patterns, and idioms that are generally applicable to writing high-quality, staff senior level code, to fix bugs.
    3. Common pitfalls and error patterns in software engineering that are relevant to identifying and fixing buggy code.
    
    Key requirements for rules:
    - DO NOT suggest testing the implementation. The agent using your ruleset is UNABLE to test its implementation. It must generate a correct patch on the first attempt.
    - DO NOT suggest reflecting on a past trajectory or attempt. The agent using your ruleset is UNABLE to reflect on a past trajectory or attempt. It must generate a correct patch on the first attempt.
    - Generated rules must be concise (less than 80 words) and should be focused on a single, specific aspect or insight.
    - Generated rules must be unique with respect to other, existing rules and contribute a new, unique piece of information, knowledge or perspective.

    This ruleset should serve as the basis for guiding future task attempts in locating and fixing bugs to a successful completion.
    It should empower the agent to improve its planning, reasoning, coding, and bug localization skills.

    """

DISTILLATION_FROM_SUCCESSFUL_TRAJECTORIES_REPOSITORY_LEVEL_PROMPT_TEMPLATE = """
    You are given a set of successful task trajectories that relate to fixing issues the real-world repository '{repository_name}'. 
    During these trajectories you correctly identified the location of the buggy code, wrote
    a patch which fixed the bug in the code and passed all test cases, meaning you also didn't in introduce any new bugs. 
    Due to the natural evolution of software over time 
    the state of the repository when you carried out the tasks in the example trajectories below may differ slightly. 
    You might encounter differences with respect to the project structure, and file, class, method or variable names. 
    If you encounter conflicting information, do not record any rules regarding the conflicting elements.

    Below follow the past successful task trajectories. The set of trajectories is delimited by the <PAST_SUCCESSFUL_TRAJECTORIES> and </PAST_SUCCESSFUL_TRAJECTORIES> tags. 
    Each trajectory is wrapped by the <TRAJECTORY-i> and </TRAJECTORY-i> tags, where i identifies the i-th trajectory in the set below:
    <PAST_SUCCESSFUL_TRAJECTORIES>
{past_successful_trajectories}
    </PAST_SUCCESSFUL_TRAJECTORIES>

    Next, follows the frozen set of high-level, general purpose rules that you have extracted previously. These rules are READ-only, you must not perform any operations on them. 
    You may refer to these rules directly in the repository level rules as 'GENERAL PURPOSE RULE-i' to highlight their specific application, knowledge gaps or discrepancies with respect to the current repository:
{general_ruleset}

    Below follows the modifiable set of repository-level rules that you have extracted so far. The repository-level ruleset is limited to {ruleset_cap} rules. Any rules beyond {ruleset_cap} rules will be ignored:
{current_repository_level_ruleset}

    By examining the successful trajectories, and the existing general purpose and repository-level rules above you should update the repository-level ruleset by adding, editing, upvoting or downvoting repository-level rules. 
    The resulting ruleset must consist of repository-specific knowledge, insights or tips that are unique to this codebase and provide new insights that are distinct from the general purpose rules. 
    Repository-level rules may cover the following aspects:
    1. Repository-level bug localization and environment exploration patterns that help locate relevant code sections quickly, including key file locations, module relationships.
    2. Repository-level coding conventions, architectural principles, design patterns, and implementation approaches that are consistently used across the codebase and should be followed when making changes.
    3. Repository-level error or exception handling strategies, including custom errors or exceptions
    4. The application domain of the project (e.g., Does the software handle images or text and what kind? Is it a command line application or does it have a GUI? Does it handle HTTP requests? Is it a highly technical, mathematical application?)
    5. Common edge cases or failure modes related to the project's specific application domain  What are common errors or potential pitfalls in these application domains?).
    
    Key requirements for rules:
    - DO NOT suggest testing the implementation. The agent must generate correct patches on the first attempt by leveraging general and repository-specific rules identified above.
    - Generated rules must be concise (less than 80 words) and should be focused on a single, specific aspect or insight.
    - Generated rules must be unique with respect to other, existing rules and contribute a new, unique piece of information, knowledge or perspective.

    This ruleset serves as the basis for guiding future task attempts within this repository in locating and fixing bugs to a successful completion.
    It should empower the agent to improve its planning, reasoning, coding, and bug localization skills.
    
{remaining_slots_information}
    """
    
DISTILLATION_FROM_SET_OF_SUCCESSFUL_TRAJECTORIES_REPOSITORY_LEVEL_PROMPT_TEMPLATE = """
    Below you will find multiple past attempts at fixing a bug in an open-source code repository. The first few trajectories show failed attempts, the last trajectory shows a successful bug fix.
    All attempts are related to fixing the same bug in the same codebase.
    Compare and contrast the successful and failed attempts to understand why the initial attempts failed and 
    which change in the reasoning, planning, coding or bug localization strategy could have led to a correct patch generation in the first attempt.
    Consider the self-reflections that took place between the failed attempts to understand which changes were made in the reasoning, planning, coding or bug localization strategy that
    led to the bug being fixed in the last trajectory.

    Below follow the task attempts denoted by <FAILED_TASK_ATTEMPT-i> and </FAILED_TASK_ATTEMPT-i> tags where i identifies the i-th failed attempt and the 
    successful task attempt is denoted by the <SUCCESSFUL_TASK_ATTEMPT> and </SUCCESSFUL_TASK_ATTEMPT> tags. Only failed task attempts contain a self-reflection:
{success_failure_trajectory}

    Next, follows the frozen set of high-level, general purpose rules that you have extracted previously. These rules are READ-only, you must not perform any operations on them. 
    You may refer to these rules directly in the repository level rules as 'GENERAL PURPOSE RULE-i' to highlight their specific application, knowledge gaps or discrepancies with respect to the current repository:
{general_ruleset}

    Below follows the modifiable set of repository-level rules that you have extracted so far. The repository-level ruleset is limited to {ruleset_cap} rules. Any rules beyond {ruleset_cap} rules will be ignored:
{current_repository_level_ruleset}

    By examining and comparing the successful and failed attempts, 
    and the existing general purpose and repository-level rules above you should update the repository-level ruleset by adding, editing, upvoting or downvoting repository-level rules. 
    The resulting ruleset must consist of repository-specific knowledge, insights or tips that are unique to this repository and provide new insights that are distinct from the general purpose rules. 
    Repository-level rules may cover the following aspects:
    1. Repository-level bug localization and environment exploration patterns that help locate relevant code sections quickly, including key file locations, module relationships.
    2. Repository-level coding conventions, architectural principles, design patterns, and implementation approaches that are consistently used across the codebase and should be followed when making changes.
    3. Repository-level error or exception handling strategies, including custom errors or exceptions
    4. The application domain of the project (e.g., Does the software handle images or text and what kind? Is it a command line application or does it have a GUI? Does it handle HTTP requests? Is it a highly technical, mathematical application?)
    5. Common edge cases or failure modes related to the project's specific application domain  What are common errors or potential pitfalls in these application domains?).
    
    Key requirements for rules:
    - DO NOT suggest testing the implementation. The agent must generate correct patches on the first attempt by leveraging general and repository-specific rules identified above.
    - DO NOT suggest reflecting on a past trajectory or attempt. The agent using your ruleset is UNABLE to reflect on a past trajectory or attempt. It must generate a correct patch on the first attempt.
    - Generated rules must be concise (less than 80 words) and should be focused on a single, specific aspect or insight.
    - Generated rules must be unique with respect to other, existing rules and contribute a new, unique piece of information, knowledge or perspective.

    This ruleset serves as the basis for guiding future task attempts within this repository in locating and fixing bugs to a successful completion.
    It should empower the agent to improve its planning, reasoning, coding, and bug localization skills.

{remaining_slots_information}
    """

RULESET_CAP_WARNING = """You have reached the maximum ruleset size of {ruleset_cap}. The ADD operation is now INVALID.
To reduce the ruleset size, prune low-utility rules that overlap with others by performing the DOWNVOTE operation on them."""

REMAINING_SLOTS_INFORMATION = """You may add up to {remaining_slots} more rules to the ruleset before reaching the maximum of {ruleset_cap} rules."""


def main(should_distill_general_knowledge, should_distill_repository_level_knowledge):
    llm_client = ... # TODO add your LLM client here

    response_format = {
        "type": "json",
        "schemaName": "json_response",
        "schema": {
            "type": "object",
            "properties": {
                "operations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "operation_type": {
                                "type": "string",
                                "enum": ["ADD", "EDIT", "UPVOTE", "DOWNVOTE"]
                            },
                            "rule_id": {"type": "integer"},
                            "rule_content": {"type": "string"}
                        },
                        "required": ["operation_type"],
                        "additionalProperties": False
                    },
                }
            },
            "required": ["operations"],
            "additionalProperties": False
        }
    }

    response_format_repo_specific = {
        "type": "json",
        "schemaName": "json_response",
        "schema": {
            "type": "object",
            "properties": {
                "operations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "operation_type": {
                                "type": "string",
                                "enum": ["ADD", "EDIT", "UPVOTE", "DOWNVOTE"]
                            },
                            "rule_id": {"type": "integer"},
                            "rule_content": {"type": "string"},
                            "knowledge_type": {
                                "type": "string",
                                "enum": [
                                    "repository_structure",
                                    "architectural_pattern",
                                    "coding_convention",
                                    "application_domain",
                                    "error_pattern"
                                ]
                            }
                        },
                        "required": ["operation_type", 'knowledge_type'],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["operations"],
            "additionalProperties": False
        }
    }

    path_to_ruleset = 'data/ruleset.json'
    # To test persistence across runs incl UP/DOWNVOTE and EDIT operation types
    if os.path.exists(path_to_ruleset):
        with open(path_to_ruleset, 'r') as f:
            current_ruleset = json.load(f)
    else:
        current_ruleset = []

    costs_for_knowledge_distillation = {
        "runs": 0,
        "dollars_spent": 0
    }
    operation_log = {
        'sets_of_successful_trajectories': {'statistics': {}, 'logs': []},
        'success_failure_pairs': {'statistics': {}, 'logs': []},
        'sets_of_successful_in_repo_trajectories': {'statistics': {}, 'logs': []},
        'success_failure_pairs_repository_level': {'statistics': {}, 'logs': []}
    }

    all_past_successful_trajectories = json.load(open('data/all_successful_patch_trajectories.json', encoding='utf-8'))
    samples_with_success_failure_pairs = [t for t in all_past_successful_trajectories if t['failed_attempts']]

    seed(42)
    shuffle(all_past_successful_trajectories)
    shuffle(samples_with_success_failure_pairs)
        
    RULESET_CAP = math.ceil(math.sqrt(len(all_past_successful_trajectories))) # For ablations I can do e.g. *2, *1/2
    
    if should_distill_general_knowledge:
        django_trajectories = [t for t in all_past_successful_trajectories if 'django' in t['instance_id']]
        other_trajectories = [t for t in all_past_successful_trajectories if 'django' not in t['instance_id']]

        shuffle(django_trajectories)
        shuffle(other_trajectories)

        # Run sets of successful trajectories distillation scenario
        # Process all trajectories from both lists, continuing to sample from the smaller list
        # after it's been exhausted to fully process the larger list
        larger_list, smaller_list = (django_trajectories, other_trajectories) if len(django_trajectories) >= len(other_trajectories) else (other_trajectories, django_trajectories)
        smaller_list_length = len(smaller_list)
        larger_list_length = len(larger_list)
        
        # Process all trajectories once
        for i in range(larger_list_length):
            # For the first part, take one from each list until smaller list is exhausted
            if i < smaller_list_length:
                samples = [larger_list[i], smaller_list[i]]
            # After smaller list is exhausted, continue with larger list and randomly sample from smaller list
            else:
                # Randomly sample from the smaller list
                random_index = randint(0, smaller_list_length - 1)
                samples = [larger_list[i], smaller_list[random_index]]

            formatted_samples_for_prompt = []
            for j, sample in enumerate(samples):
                messages = sample['patch_generation_trajectory']
                
                formatted_trajectory = ''
                for message in messages:
                    formatted_trajectory += f"{message['role'].upper()}: {message['content']}\n"

                formatted_samples_for_prompt.append(
                    f"<TRAJECTORY-{j}>\n"
                    f"{formatted_trajectory}"
                    f"\n</TRAJECTORY-{j}>"
                )

            ruleset_for_prompt = [{k: v for k, v in rule.items() if k != 'importance'} for rule in current_ruleset]
            remaining_slots = RULESET_CAP - len(current_ruleset)

            distill_from_set_of_successful_trajectories_prompt = DISTILLATION_FROM_SUCCESSFUL_TRAJECTORIES_PROMPT_TEMPLATE.format(
                current_ruleset=indent('\n'.join(f"Rule {rule['id']}: {rule['content']}" for rule in
                                                 ruleset_for_prompt) if ruleset_for_prompt else "No rules extracted yet, the only valid operation is thus 'ADD'.",
                                       '\t'),
                past_successful_trajectories=indent('\n'.join(formatted_samples_for_prompt), '\t'),
                remaining_slots_information=indent(REMAINING_SLOTS_INFORMATION.format(remaining_slots=remaining_slots, ruleset_cap=RULESET_CAP) if remaining_slots > 0 else RULESET_CAP_WARNING.format(ruleset_cap=RULESET_CAP), '\t'),
                ruleset_cap=RULESET_CAP
            )

            operation_explaination_prompt = OPERATION_EXPLANATION_PROMPT_TEMPLATE.format(
                ruleset_indices_hint=f" and in [0, {len(current_ruleset) - 1}]" if len(current_ruleset) > 0 else "")

            chat = Chat() \
                .add_system(SYSTEM_PROMPT) \
                .add_user(distill_from_set_of_successful_trajectories_prompt + operation_explaination_prompt)
            try:
                response = prompt_o1_model_with(llm_client, chat, response_format)
            except Exception as e:
                print(f"Error: {e}.\nRetrying with Claude Sonnet 3.7 ...")
                try:
                    response = prompt_sonnet_37_model_with(llm_client, chat)
                except Exception as e:
                    print(f"Error: {e}.\nRecovery attempt failed. Skipping this distillation sample.")
                    continue

            costs_for_knowledge_distillation = update_costs(response, costs_for_knowledge_distillation)

            pattern = r'{[\s\S]*}'
            result = re.search(pattern, response.content, re.DOTALL)
            print(f"========== SETS OF SUCCESSES - ITERATION {i} ==============")
            print("This call cost {} USD. You have used {} credits ({} USD) out of the maximum {}"
                  .format(round(float(response.spent.amount) / 100000, 4), response.updated.current.amount,
                          round(float(response.updated.current.amount) / 100000, 4), response.updated.maximum.amount))
            try:
                operations_to_perform = ast.literal_eval(result.group(0))['operations']
                operation_log['sets_of_successful_trajectories']['logs'].append(
                    {'prompt': distill_from_set_of_successful_trajectories_prompt + operation_explaination_prompt,
                     'operations': deepcopy(operations_to_perform)})
                current_ruleset = process_operations(current_ruleset, operations_to_perform, ruleset_cap=RULESET_CAP)
            except Exception:
                print("Failed to extract knowledge from this sample. Model didnt adhere to response format!")

        # Run success/failure pairs of trajectories distillation scenario
        for m, sample in enumerate(samples_with_success_failure_pairs):
            failed_attempts = sample['failed_attempts']

            trajectory_context = ''
            for i, failed_attempt in enumerate(failed_attempts):
                formatted_trajectory = ''
                messages = failed_attempt['patch_data']
                for message in messages:
                    formatted_trajectory += f"{message['role'].upper()}: {message['content']}\n"
                trajectory_context += (
                    f"<FAILED_TASK_ATTEMPT-{i}>\n"
                    f'{formatted_trajectory}'
                    f"</FAILED_TASK_ATTEMPT-{i}>\n"
                )
                
            formatted_trajectory = ''
            messages = sample['patch_generation_trajectory']
            for message in messages:
                formatted_trajectory += f"{message['role'].upper()}: {message['content']}\n"
            trajectory_context += (
                f"<SUCCESSFUL_TASK_ATTEMPT>\n"
                f'{formatted_trajectory}'
                f"</SUCCESSFUL_TASK_ATTEMPT>\n"
            )

            ruleset_for_prompt = [{k: v for k, v in rule.items() if k != 'importance'} for rule in current_ruleset]
            remaining_slots = RULESET_CAP - len(current_ruleset)

            distill_from_success_failure_pairs_prompt = DISTILLATION_FROM_SUCCESS_FAILURE_PAIRS_PROMPT_TEMPLATE.format(
                current_ruleset=indent('\n'.join(f"Rule {rule['id']}: {rule['content']}" for rule in
                                                 ruleset_for_prompt) if ruleset_for_prompt else "No rules extracted yet, the only valid operation is thus 'ADD'.",
                                       '\t'),
                success_failure_trajectory=indent(trajectory_context, '\t'),
                remaining_slots_information=indent(REMAINING_SLOTS_INFORMATION.format(remaining_slots=remaining_slots, ruleset_cap=RULESET_CAP) if remaining_slots > 0 else RULESET_CAP_WARNING.format(ruleset_cap=RULESET_CAP), '\t'),
                ruleset_cap=RULESET_CAP
            )

            operation_explaination_prompt = OPERATION_EXPLANATION_PROMPT_TEMPLATE.format(
                ruleset_indices_hint=f" and in [0, {len(current_ruleset) - 1}]" if len(current_ruleset) > 0 else "")

            chat = Chat() \
                .add_system(SYSTEM_PROMPT) \
                .add_user(distill_from_success_failure_pairs_prompt + operation_explaination_prompt)
            try:
                response = prompt_o1_model_with(llm_client, chat, response_format)
            except Exception as e:
                print(f"Error: {e}.\nRetrying with Claude Sonnet 3.7 ...")
                try:
                    response = prompt_sonnet_37_model_with(llm_client, chat)
                except Exception as e:
                    print(f"Error: {e}.\nRecovery attempt failed. Skipping this distillation sample.")
                    continue

            costs_for_knowledge_distillation = update_costs(response, costs_for_knowledge_distillation)

            # Crutch for models that cannot respond in JSON directly, must extract the JSON object via regex
            pattern = r'{[\s\S]*}'
            result = re.search(pattern, response.content, re.DOTALL)
            print(f"========== SUCCESS-FAILURE PAIRS - ITERATION {m} ==============")
            print("This call cost {} USD. You have used {} credits ({} USD) out of the maximum {}"
                  .format(round(float(response.spent.amount) / 100000, 4), response.updated.current.amount,
                          round(float(response.updated.current.amount) / 100000, 4), response.updated.maximum.amount))
            try:
                operations_to_perform = ast.literal_eval(result.group(0))['operations']
                operation_log['success_failure_pairs']['logs'].append(
                    {'prompt': distill_from_success_failure_pairs_prompt + operation_explaination_prompt,
                     'operations': deepcopy(operations_to_perform)})
                current_ruleset = process_operations(current_ruleset, operations_to_perform, ruleset_cap=RULESET_CAP)

            except Exception:
                print("Failed to extract knowledge from this sample. Model didnt adhere to response format!")

        with open(path_to_ruleset, 'w+') as f:
            json.dump(current_ruleset, f)

    if should_distill_repository_level_knowledge:
        available_repositories = {sample['instance_id'].split('__')[0] for sample in all_past_successful_trajectories}
        samples_grouped_by_repository = {repository: [] for repository in available_repositories}
        
        path_to_repository_level_ruleset = 'data/repo_ruleset.json'
        # To test persistence across runs incl UP/DOWNVOTE and EDIT operation types
        if os.path.exists(path_to_repository_level_ruleset):
            with open(path_to_repository_level_ruleset, 'r') as f:
                current_repository_level_ruleset = json.load(f)
        else:
            current_repository_level_ruleset = deepcopy(samples_grouped_by_repository)

        for sample in all_past_successful_trajectories:
            samples_grouped_by_repository[sample['instance_id'].split('__')[0]].append(sample)
        
        # Run sets of successful trajectories for repository-level distillation scenario
        repositories_with_initialized_ruleset_caps = []
        for i, (repository, samples) in enumerate(samples_grouped_by_repository.items()):
            set_size = 2 if len(samples) > 1 else 1

            repository_ruleset_cap = math.ceil(math.sqrt(len(samples)))
            if repository not in repositories_with_initialized_ruleset_caps:
                for success_failure_pair in [s for s in samples_with_success_failure_pairs if s['repository'] == repository]:
                    success_failure_pair['ruleset_cap'] = repository_ruleset_cap
                repositories_with_initialized_ruleset_caps.append(repository)

            are_samples_processed = False
            for j in range(0, len(samples), set_size):
                # If our total amount of samples are odd, we need to include three items in the last set
                if abs(j * set_size - len(samples)) == 1 or set_size == 1:
                    samples_in_set = samples[j:]
                    are_samples_processed = True
                elif not are_samples_processed:
                    samples_in_set = samples[j:j + set_size]
                else:
                    break

                formatted_samples_for_prompt = []
                for k, sample in enumerate(samples_in_set):
                    messages = sample['patch_generation_trajectory']

                    formatted_trajectory = ''
                    for message in messages:
                        formatted_trajectory += f"{message['role'].upper()}: {message['content']}\n"

                    formatted_samples_for_prompt.append(
                        f"<TRAJECTORY-{k}>\n"
                        f"{formatted_trajectory}"
                        f"\n</TRAJECTORY-{k}>"
                    )

                general_ruleset_for_prompt = [{k: v for k, v in rule.items() if k != 'importance'} for rule in current_ruleset]
                repository_level_ruleset_for_prompt = [{k: v for k, v in rule.items() if k != 'importance'} for rule in current_repository_level_ruleset[repository]]
                remaining_slots = repository_ruleset_cap - len(current_repository_level_ruleset[repository])

                distill_from_set_of_successful_trajectories_repository_level_prompt = DISTILLATION_FROM_SUCCESSFUL_TRAJECTORIES_REPOSITORY_LEVEL_PROMPT_TEMPLATE.format(
                    repository_name=repository,
                    general_ruleset=indent('\n'.join(f"Rule {rule['id']}: {rule['content']}" for rule in
                                                     general_ruleset_for_prompt) if general_ruleset_for_prompt
                                           else "No general rules extracted yet, the only valid operation is thus 'ADD'.",
                                           '\t'),
                    current_repository_level_ruleset=indent(
                        '\n'.join(f"Rule {rule['id']} of type {rule['knowledge_type']}: {rule['content']}" for rule in
                                  repository_level_ruleset_for_prompt) if repository_level_ruleset_for_prompt
                        else "No repository-level rules extracted yet, the only valid operation is thus 'ADD'.",
                        '\t'),
                    past_successful_trajectories=indent('\n'.join(formatted_samples_for_prompt), '\t'),
                    remaining_slots_information=indent(REMAINING_SLOTS_INFORMATION.format(remaining_slots=remaining_slots, ruleset_cap=repository_ruleset_cap) if remaining_slots > 0 else RULESET_CAP_WARNING.format(ruleset_cap=repository_ruleset_cap), '\t'),
                    ruleset_cap=repository_ruleset_cap
                )

                operation_explaination_prompt = OPERATION_EXPLANATION_REPOSITORY_LEVEL_PROMPT_TEMPLATE.format(
                    ruleset_indices_hint=f" and in [0, {len(current_repository_level_ruleset[repository]) - 1}]" if len(current_repository_level_ruleset[repository]) > 0 else "")

                chat = Chat() \
                    .add_system(SYSTEM_PROMPT) \
                    .add_user(distill_from_set_of_successful_trajectories_repository_level_prompt + operation_explaination_prompt)
                try:
                    response = prompt_o1_model_with(llm_client, chat, response_format_repo_specific)
                except Exception as e:
                    print(f"Error: {e}.\nRetrying with Claude Sonnet 3.7 ...")
                    try:
                        response = prompt_sonnet_37_model_with(llm_client, chat) 
                    except Exception as e:
                        print(f"Error: {e}.\nRecovery attempt failed. Skipping this distillation sample.")
                        continue

                costs_for_knowledge_distillation = update_costs(response, costs_for_knowledge_distillation)

                pattern = r'{[\s\S]*}'
                result = re.search(pattern, response.content, re.DOTALL)
                print(f"========== SETS OF SUCCESSES - {repository} - ITERATION {i}.{j // set_size} ==============")
                print("This call cost {} USD. You have used {} credits ({} USD) out of the maximum {}"
                        .format(round(float(response.spent.amount) / 100000, 4), response.updated.current.amount,
                              round(float(response.updated.current.amount) / 100000, 4), response.updated.maximum.amount))
                try:
                    operations_to_perform = ast.literal_eval(result.group(0))['operations']
                    operation_log['sets_of_successful_in_repo_trajectories']['logs'].append(
                        {'prompt': distill_from_set_of_successful_trajectories_repository_level_prompt + operation_explaination_prompt,
                         'operations': deepcopy(operations_to_perform)})

                    current_repository_level_ruleset = process_operations(current_repository_level_ruleset, operations_to_perform,
                                                         ruleset_cap=repository_ruleset_cap, is_ruleset_repository_level=True, repository=repository)
                except Exception:
                    print("Failed to extract knowledge from this sample. Model didnt adhere to response format!")
                    
        # Run success/failure pairs of trajectories distillation scenario
        for m, sample in enumerate(samples_with_success_failure_pairs):
            failed_attempts = sample['failed_attempts']

            trajectory_context = ''
            for i, failed_attempt in enumerate(failed_attempts):
                formatted_trajectory = ''
                messages = failed_attempt['patch_data']
                for message in messages:
                    formatted_trajectory += f"{message['role'].upper()}: {message['content']}\n"
                trajectory_context += (
                    f"<FAILED_TASK_ATTEMPT-{i}>\n"
                    f'{formatted_trajectory}'
                    f"</FAILED_TASK_ATTEMPT-{i}>\n"
                )
                
            formatted_trajectory = ''
            messages = sample['patch_generation_trajectory']
            for message in messages:
                formatted_trajectory += f"{message['role'].upper()}: {message['content']}\n"
            trajectory_context += (
                f"<SUCCESSFUL_TASK_ATTEMPT>\n"
                f'{formatted_trajectory}'
                f"</SUCCESSFUL_TASK_ATTEMPT>\n"
            )

            general_ruleset_for_prompt = [{k: v for k, v in rule.items() if k != 'importance'} for rule in current_ruleset]
            repository_level_ruleset_for_prompt = [{k: v for k, v in rule.items() if k != 'importance'} for rule in current_repository_level_ruleset[sample['repository']]]
            remaining_slots = sample['ruleset_cap'] - len(current_repository_level_ruleset[sample['repository']])

            distill_from_success_failure_pairs_prompt = DISTILLATION_FROM_SET_OF_SUCCESSFUL_TRAJECTORIES_REPOSITORY_LEVEL_PROMPT_TEMPLATE.format(
                repository_name=sample['repository'],
                general_ruleset=indent('\n'.join(f"Rule {rule['id']}: {rule['content']}" for rule in
                                                 general_ruleset_for_prompt) if general_ruleset_for_prompt else "No general rules extracted yet, the only valid operation is thus 'ADD'.",
                                       '\t'),
                current_repository_level_ruleset=indent(
                        '\n'.join(f"Rule {rule['id']} of type {rule['knowledge_type']}: {rule['content']}" for rule in
                                  repository_level_ruleset_for_prompt) if repository_level_ruleset_for_prompt
                        else "No repository-level rules extracted yet, the only valid operation is thus 'ADD'.",
                        '\t'),
                success_failure_trajectory=indent(trajectory_context, '\t'),
                
                remaining_slots_information=indent(REMAINING_SLOTS_INFORMATION.format(remaining_slots=remaining_slots, ruleset_cap=sample['ruleset_cap']) if remaining_slots > 0 else RULESET_CAP_WARNING.format(ruleset_cap=sample['ruleset_cap']), '\t'),
                ruleset_cap=sample['ruleset_cap']
            )

            operation_explaination_prompt = OPERATION_EXPLANATION_REPOSITORY_LEVEL_PROMPT_TEMPLATE.format(
                ruleset_indices_hint=f" and in [0, {len(current_repository_level_ruleset[sample['repository']]) - 1}]" if len(current_repository_level_ruleset[sample['repository']]) > 0 else "")

            chat = Chat() \
                .add_system(SYSTEM_PROMPT) \
                .add_user(distill_from_success_failure_pairs_prompt + operation_explaination_prompt)
            try:
                response = prompt_o1_model_with(llm_client, chat, response_format)
            except Exception as e:
                print(f"Error: {e}.\nRetrying with Claude Sonnet 3.7 ...")
                try:
                    response = prompt_sonnet_37_model_with(llm_client, chat)
                except Exception as e:
                    print(f"Error: {e}.\nRecovery attempt failed. Skipping this distillation sample.")
                    continue

            costs_for_knowledge_distillation = update_costs(response, costs_for_knowledge_distillation)

            # Crutch for models that cannot respond in JSON directly, must extract the JSON object via regex
            pattern = r'{[\s\S]*}'
            result = re.search(pattern, response.content, re.DOTALL)
            print(f"========== SUCCESS-FAILURE PAIRS - REPOSITORY LEVEL - ITERATION {m} ==============")
            print("This call cost {} USD. You have used {} credits ({} USD) out of the maximum {}"
                  .format(round(float(response.spent.amount) / 100000, 4), response.updated.current.amount,
                          round(float(response.updated.current.amount) / 100000, 4), response.updated.maximum.amount))
            try:
                operations_to_perform = ast.literal_eval(result.group(0))['operations']
                operation_log['success_failure_pairs_repository_level']['logs'].append(
                    {'prompt': distill_from_success_failure_pairs_prompt + operation_explaination_prompt,
                        'operations': deepcopy(operations_to_perform)})

                current_repository_level_ruleset = process_operations(current_repository_level_ruleset, operations_to_perform,
                                                        ruleset_cap=sample['ruleset_cap'], is_ruleset_repository_level=True, repository=sample['repository'])
            except Exception as e:
                print(f"Error: {e}.\nRecovery attempt failed. Skipping this distillation sample.")
                continue

        with open(path_to_repository_level_ruleset, 'w+') as f:
            json.dump(current_repository_level_ruleset, f)

    # Get operation statistics for sets of successful trajectories
    sets_operations = dict(ADD=0, EDIT=0, UPVOTE=0, DOWNVOTE=0)
    for operations in operation_log['sets_of_successful_trajectories']['logs']:
        for operation in operations['operations']:
            sets_operations[operation['operation_type']] += 1

    # Get operation statistics for success failure pairs
    pairs_operations = dict(ADD=0, EDIT=0, UPVOTE=0, DOWNVOTE=0)
    for operations in operation_log['success_failure_pairs']['logs']:
        for operation in operations['operations']:
            pairs_operations[operation['operation_type']] += 1

    # Update the operation_log statistics
    operation_log['sets_of_successful_trajectories']['statistics'] = sets_operations
    operation_log['success_failure_pairs']['statistics'] = pairs_operations

    with open('data/operation_log.json', 'w+') as f:
        json.dump(operation_log, f)

    with open('data/costs_for_knowledge_distillation.json', 'w+') as f:
        json.dump(costs_for_knowledge_distillation, f)


def prompt_o1_model_with(llm_client, chat, response_format):
    return llm_client.chat(
        chat=chat,
        profile=Profile.OPENAI_O_1,
        parameters={
            LLMParameters.ResponseFormat: Parameters.JsonValue(response_format),
        }
    )
    
def prompt_sonnet_37_model_with(llm_client, chat):
    return llm_client.chat(
        chat=chat,
        profile=Profile.ANTHROPIC_CLAUDE_37_SONNET,
        parameters={
            LLMParameters.Temperature: Parameters.FloatValue(0),
        }
    )


def update_costs(response, costs_for_knowledge_distillation):
    costs_for_knowledge_distillation['runs'] += 1
    costs_for_knowledge_distillation['dollars_spent'] += float(response.spent.amount) / 100000
    return costs_for_knowledge_distillation


def process_operations(current_ruleset, operations_to_perform, ruleset_cap:int, is_ruleset_repository_level: bool = False, repository:str=None):
    if is_ruleset_repository_level:
        current_repository_level_ruleset = current_ruleset
        current_ruleset = current_ruleset[repository]


    while operations_to_perform:
        operation = operations_to_perform.pop()

        match operation['operation_type']:
            case "ADD":
                if len(current_ruleset) < ruleset_cap:
                    if not is_ruleset_repository_level:
                        current_ruleset.append({
                            "id": len(current_ruleset),
                            "content": operation['rule_content'],
                            "importance": 4
                        })
                    else:
                        current_ruleset.append({
                            "id": len(current_ruleset),
                            "content": operation['rule_content'],
                            "importance": 4,
                            "knowledge_type": operation['knowledge_type']
                        })
            case "EDIT":
                try:
                    current_ruleset[operation['rule_id']]['content'] = operation['rule_content']
                    current_ruleset[operation['rule_id']]['importance'] += 1
                except IndexError:
                    print(
                        f'Rule_ID {operation["rule_id"]} does not exist in the current ruleset of size {len(current_ruleset)}.')
            case "UPVOTE":
                try:
                    current_ruleset[operation['rule_id']]['importance'] += 1
                except IndexError:
                    print(
                        f'Rule_ID {operation["rule_id"]} does not exist in the current ruleset of size {len(current_ruleset)}.')
            case "DOWNVOTE":
                try:
                    current_ruleset[operation['rule_id']]['importance'] -= 1
                    if current_ruleset[operation['rule_id']]['importance'] == 0:
                        removed_rule = current_ruleset.pop(operation['rule_id'])
                        # All operations with a rule_id greater than the removed rule's id need to be updated
                        for operation_to_update in operations_to_perform:
                            if operation_to_update['rule_id'] > removed_rule['id']:
                                operation_to_update['rule_id'] -= 1

                        # Update the indices of the ruleset by shifting all indices greater than the removed rule's id by -1
                        current_ruleset = update_ruleset_indices(current_ruleset)
                except IndexError:
                    print(
                        f'Rule_ID {operation["rule_id"]} does not exist in the current ruleset of size {len(current_ruleset)}.')

    if is_ruleset_repository_level:
        current_repository_level_ruleset[repository] = current_ruleset
        return current_repository_level_ruleset
    else:
        return current_ruleset

def update_ruleset_indices(current_ruleset):
    for i, rule in enumerate(current_ruleset):
        rule['id'] = i
    return current_ruleset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process command line arguments.")
    parser.add_argument('--general', action='store_true', help="Distil general, high-level knowledge.")
    parser.add_argument('--repository', action='store_true', help="Distil repository-level knowledge.")

    args = parser.parse_args()

    should_distill_general_knowledge = args.general
    should_distill_repository_level_knowledge = args.repository

    main(should_distill_general_knowledge, should_distill_repository_level_knowledge)
