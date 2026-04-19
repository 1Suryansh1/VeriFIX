from __future__ import annotations

from collections import Counter

from verifix.edit_dsl.operators import (
    delete_statement_operator,
    get_candidate_edits,
    insert_statement_operator,
    negate_condition_operator,
    replace_comparison_operator,
    replace_literal_operator,
    rewrite_call_argument_operator,
    rewrite_len_call_operator,
    rewrite_range_call_stop_operator,
    rewrite_for_iter_slice_operator,
    rewrite_assign_subscript_target_operator,
    rewrite_assign_to_max_operator,
    rewrite_return_plus_one_call_operator,
    rewrite_return_listcomp_prefix_operator,
    rewrite_subscript_index_operator,
    swap_operands_operator,
    wrap_condition_operator,
)
from verifix.parser.ast_builder import build_ast


BUGGY_SOURCE = """
def find_max(arr):
    max_val = arr[0]
    for i in range(1, len(arr)):
        if arr[i] < max_val:   # BUG: should be >
            max_val = arr[i]
    return max_val
"""


def _buggy_ast():
    return build_ast(BUGGY_SOURCE, "src/find_max.py", language="python")


def test_replace_comparison_operator_contains_gt_replacement() -> None:
    annotated = _buggy_ast()
    compare_node = next(node for node in annotated.find_by_type("Compare") if node.lineno == 5)

    edits = replace_comparison_operator(compare_node, annotated)

    assert any(edit.replacement_text == "arr[i] > max_val" for edit in edits)


def test_negate_condition_operator_wraps_with_not() -> None:
    annotated = _buggy_ast()
    compare_node = next(node for node in annotated.find_by_type("Compare") if node.lineno == 5)

    edits = negate_condition_operator(compare_node, annotated)

    assert len(edits) == 1
    assert edits[0].replacement_text == "not (arr[i] < max_val)"


def test_replace_literal_operator_for_zero_contains_one_and_minus_one() -> None:
    annotated = _buggy_ast()
    zero_node = next(
        node
        for node in annotated.find_by_type("Constant")
        if node.source_text.strip() == "0"
    )

    edits = replace_literal_operator(zero_node, annotated)
    replacements = {edit.replacement_text for edit in edits}

    assert "1" in replacements
    assert "-1" in replacements


def test_replace_literal_operator_supports_empty_list_literal_upgrade() -> None:
    src = "def subsequences(a):\n    return []\n"
    annotated = build_ast(src, "src/subsequences.py")
    list_node = next(node for node in annotated.find_by_type("List") if node.source_text.strip() == "[]")

    edits = replace_literal_operator(list_node, annotated)

    assert any(edit.replacement_text == "[[]]" for edit in edits)


def test_delete_statement_operator_not_applied_to_functiondef() -> None:
    annotated = _buggy_ast()
    function_node = annotated.find_by_type("FunctionDef")[0]

    edits = delete_statement_operator(function_node, annotated)

    assert edits == []


def test_swap_operands_operator_returns_no_edits_for_identical_operands() -> None:
    src = "def f(a):\n    return a + a\n"
    annotated = build_ast(src, "src/identical.py")
    binop_node = annotated.find_by_type("BinOp")[0]

    edits = swap_operands_operator(binop_node, annotated)

    assert edits == []


def test_get_candidate_edits_respects_suspicious_lines() -> None:
    annotated = _buggy_ast()

    edits = get_candidate_edits(annotated, suspicious_lines=[5])

    assert edits
    assert all(edit.line_number == 5 for edit in edits)


def test_get_candidate_edits_respects_max_edits_per_node() -> None:
    annotated = _buggy_ast()

    edits = get_candidate_edits(annotated, suspicious_lines=[5], max_edits_per_node=1)
    counts = Counter(edit.node_id for edit in edits)

    assert counts
    assert all(count <= 1 for count in counts.values())


def test_get_candidate_edits_deduplicates_same_node_and_replacement() -> None:
    annotated = _buggy_ast()

    edits = get_candidate_edits(
        annotated,
        suspicious_lines=[5],
        max_edits_per_node=10,
        enabled_operators=["negate_condition_operator", "negate_condition_operator"],
    )

    assert len(edits) == 1
    assert edits[0].replacement_text == "not (arr[i] < max_val)"


def test_get_candidate_edits_keeps_insert_before_and_after_variants() -> None:
    src = """
def f(node, nodesvisited):
    if node in nodesvisited:
        return False
"""
    annotated = build_ast(src, "src/insert_variants.py")

    edits = get_candidate_edits(
        annotated,
        suspicious_lines=[3],
        max_edits_per_node=20,
        enabled_operators=["insert_statement_operator"],
    )

    operator_values = {edit.operator.value for edit in edits if edit.replacement_text == "nodesvisited.add(node)"}
    assert "insert_stmt_before" in operator_values
    assert "insert_stmt_after" in operator_values


def test_generated_edits_have_non_empty_node_id_and_operator() -> None:
    annotated = _buggy_ast()
    edits = get_candidate_edits(annotated, suspicious_lines=[5])

    assert edits
    assert all(bool(edit.node_id) for edit in edits)
    assert all(bool(edit.operator) for edit in edits)


def test_rewrite_subscript_index_operator_generates_plus_one_variant() -> None:
    src = "def f(arr, mid):\n    return arr[mid]\n"
    annotated = build_ast(src, "src/subscript.py")
    subscript_node = annotated.find_by_type("Subscript")[0]

    edits = rewrite_subscript_index_operator(subscript_node, annotated)
    replacements = {edit.replacement_text for edit in edits}

    assert "arr[mid + 1]" in replacements


def test_rewrite_subscript_index_operator_tuple_index_can_drop_offset() -> None:
    src = "def f(dp, i, j):\n    return dp[i - 1, j - 1]\n"
    annotated = build_ast(src, "src/tuple_subscript.py")
    subscript_node = annotated.find_by_type("Subscript")[0]

    edits = rewrite_subscript_index_operator(subscript_node, annotated)
    replacements = {edit.replacement_text for edit in edits}

    assert "dp[i - 1, j]" in replacements


def test_rewrite_call_argument_operator_can_shift_midpoint_bound() -> None:
    src = "def binsearch(start, end):\n    mid = (start + end) // 2\n    return binsearch(mid, end)\n"
    annotated = build_ast(src, "src/find_in_sorted_like.py")
    call_node = next(
        node for node in annotated.find_by_type("Call") if node.source_text.strip() == "binsearch(mid, end)"
    )

    edits = rewrite_call_argument_operator(call_node, annotated)
    replacements = {edit.replacement_text for edit in edits}

    assert "binsearch(mid + 1, end)" in replacements


def test_rewrite_call_argument_operator_can_generate_k_minus_count_for_recursive_call() -> None:
    src = """
def kth(arr, k):
    num_lessoreq = 3
    above = arr
    return kth(above, k)
"""
    annotated = build_ast(src, "src/kth_like.py")
    call_node = next(node for node in annotated.find_by_type("Call") if node.source_text.strip() == "kth(above, k)")

    edits = rewrite_call_argument_operator(call_node, annotated)
    replacements = {edit.replacement_text for edit in edits}

    assert "kth(above, k - num_lessoreq)" in replacements


def test_rewrite_call_argument_operator_supports_keyword_calls() -> None:
    src = "def f(mid, end):\n    return search(left=mid, right=end)\n"
    annotated = build_ast(src, "src/keyword_call.py")
    call_node = annotated.find_by_type("Call")[0]

    edits = rewrite_call_argument_operator(call_node, annotated)
    replacements = {edit.replacement_text for edit in edits}

    assert any(text.startswith("search(") and "left=mid" in text and "right=end" in text for text in replacements)


def test_rewrite_call_argument_operator_can_swap_any_to_all() -> None:
    src = "def sieve(n, primes):\n    return any(n % p > 0 for p in primes)\n"
    annotated = build_ast(src, "src/sieve_like.py")
    call_node = annotated.find_by_type("Call")[0]

    edits = rewrite_call_argument_operator(call_node, annotated)
    replacements = {edit.replacement_text for edit in edits}

    assert any(text.startswith("all(") for text in replacements)


def test_rewrite_call_argument_operator_can_unwrap_single_argument_call() -> None:
    src = "def flatten(x):\n    return flatten(x)\n"
    annotated = build_ast(src, "src/flatten_like.py")
    call_node = annotated.find_by_type("Call")[0]

    edits = rewrite_call_argument_operator(call_node, annotated)
    replacements = {edit.replacement_text for edit in edits}

    assert "x" in replacements


def test_wrap_condition_operator_can_add_guard() -> None:
    src = "def f(total, coins):\n    if total < 0:\n        return 0\n"
    annotated = build_ast(src, "src/possible_change_like.py")
    compare_node = annotated.find_by_type("Compare")[0]

    edits = wrap_condition_operator(compare_node, annotated)
    replacements = {edit.replacement_text for edit in edits}

    assert "(total < 0) or not coins" in replacements


def test_insert_statement_operator_generates_add_or_append_candidates() -> None:
    src = """
def depth_first_search(node, nodesvisited):
    if node in nodesvisited:
        return
    for child in node.successors:
        depth_first_search(child, nodesvisited)
"""
    annotated = build_ast(src, "src/depth_first_search.py")
    if_node = next(node for node in annotated.find_by_type("If") if node.lineno == 3)

    edits = insert_statement_operator(if_node, annotated)
    replacement_texts = {edit.replacement_text for edit in edits}
    operators = {edit.operator.value for edit in edits}

    assert "nodesvisited.add(node)" in replacement_texts
    assert "insert_stmt_before" in operators
    assert "insert_stmt_after" in operators


def test_insert_statement_operator_can_use_outer_scope_names_in_nested_function() -> None:
    src = """
def depth_first_search(startnode, goalnode):
    nodesvisited = set()

    def search_from(node):
        if node in nodesvisited:
            return False
        return any(search_from(nextnode) for nextnode in node.successors)
"""
    annotated = build_ast(src, "src/depth_first_search_nested.py")
    return_node = next(node for node in annotated.find_by_type("Return") if node.lineno == 8)

    edits = insert_statement_operator(return_node, annotated)
    replacement_texts = {edit.replacement_text for edit in edits}

    assert "nodesvisited.add(node)" in replacement_texts


def test_rewrite_subscript_index_operator_can_swap_tuple_indices() -> None:
    src = "def f(dp, i, j):\n    return dp[i, j]\n"
    annotated = build_ast(src, "src/tuple_swap.py")
    subscript_node = annotated.find_by_type("Subscript")[0]

    edits = rewrite_subscript_index_operator(subscript_node, annotated)
    replacements = {edit.replacement_text for edit in edits}

    assert "dp[j, i]" in replacements


def test_get_candidate_edits_core_tier_excludes_synthetic_operators() -> None:
    src = "def f(x):\n    return x + 1\n"
    annotated = build_ast(src, "src/tier_core.py")

    edits = get_candidate_edits(annotated, suspicious_lines=[2], operator_tier="core")

    assert edits
    assert all(edit.metadata.get("target") != "synthetic_wrap" for edit in edits)


def test_get_candidate_edits_all_tier_can_include_synthetic_operators() -> None:
    src = "def f(x):\n    return x + 1\n"
    annotated = build_ast(src, "src/tier_all.py")

    edits = get_candidate_edits(annotated, suspicious_lines=[2], operator_tier="all")

    assert any(edit.metadata.get("target") == "synthetic_wrap" for edit in edits)


def test_rewrite_for_iter_slice_operator_keeps_loop_body() -> None:
    src = "def f(arr, k):\n    for x in arr:\n        yield x\n"
    annotated = build_ast(src, "src/kheapsort_like.py")
    for_node = annotated.find_by_type("For")[0]

    edits = rewrite_for_iter_slice_operator(for_node, annotated)
    replacements = {edit.replacement_text for edit in edits}

    assert any("for x in arr[k:]:" in text and "yield x" in text for text in replacements)


def test_rewrite_assign_subscript_target_operator_rewrites_edge_to_node_index() -> None:
    src = "def f(weight_by_edge, weight_by_node, u, v, weight):\n    weight_by_edge[u, v] = min(weight_by_node[u] + weight, weight_by_node[v])\n"
    annotated = build_ast(src, "src/shortest_paths_like.py")
    assign_stmt = annotated.find_by_type("Assign")[0]

    edits = rewrite_assign_subscript_target_operator(assign_stmt, annotated)
    replacements = {edit.replacement_text for edit in edits}

    assert "weight_by_node[v] = min(weight_by_node[u] + weight, weight_by_node[v])" in replacements


def test_rewrite_len_call_operator_can_generate_minus_one_variant() -> None:
    src = "def f(digit_list):\n    return [1] + len(digit_list) * [0] + [1]\n"
    annotated = build_ast(src, "src/next_palindrome_like.py")
    len_call = next(node for node in annotated.find_by_type("Call") if node.source_text.strip() == "len(digit_list)")

    edits = rewrite_len_call_operator(len_call, annotated)
    replacements = {edit.replacement_text for edit in edits}

    assert "(len(digit_list) - 1)" in replacements


def test_rewrite_range_call_stop_operator_adds_plus_one_to_stop_bound() -> None:
    src = "def pascal(n):\n    for c in range(0, n):\n        yield c\n"
    annotated = build_ast(src, "src/pascal_like.py")
    range_call = next(node for node in annotated.find_by_type("Call") if node.source_text.strip() == "range(0, n)")

    edits = rewrite_range_call_stop_operator(range_call, annotated)
    replacements = {edit.replacement_text for edit in edits}

    assert "range(0, n + 1)" in replacements


def test_rewrite_return_plus_one_call_operator_unwraps_recursive_case() -> None:
    src = "def levenshtein(source, target):\n    return 1 + levenshtein(source[1:], target[1:])\n"
    annotated = build_ast(src, "src/levenshtein_like.py")
    return_node = annotated.find_by_type("Return")[0]

    edits = rewrite_return_plus_one_call_operator(return_node, annotated)
    replacements = {edit.replacement_text for edit in edits}

    assert "return levenshtein(source[1:], target[1:])" in replacements


def test_rewrite_assign_to_max_operator_wraps_assignment_value() -> None:
    src = "def lis_like(length, longest):\n    longest = length + 1\n"
    annotated = build_ast(src, "src/lis_like.py")
    assign_node = annotated.find_by_type("Assign")[0]

    edits = rewrite_assign_to_max_operator(assign_node, annotated)
    replacements = {edit.replacement_text for edit in edits}

    assert "longest = max(longest, length + 1)" in replacements


def test_rewrite_return_listcomp_prefix_operator_adds_prefix_variable() -> None:
    src = "def powerset(arr):\n    if arr:\n        first, *rest = arr\n        rest_subsets = powerset(rest)\n        return [[first] + subset for subset in rest_subsets]\n"
    annotated = build_ast(src, "src/powerset_like.py")
    return_node = annotated.find_by_type("Return")[-1]

    edits = rewrite_return_listcomp_prefix_operator(return_node, annotated)
    replacements = {edit.replacement_text for edit in edits}

    assert "return rest_subsets + [[first] + subset for subset in rest_subsets]" in replacements
