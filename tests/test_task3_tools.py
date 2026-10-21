"""Offline tests for the Task 3 tools."""

from tools import calculator, file_reader


def calc(expr: str) -> str:
    return calculator.invoke({"expression": expr})


def read(name: str) -> str:
    return file_reader.invoke({"file_name": name})


def test_calculator_basic_and_functions():
    assert calc("sqrt(144) + 2**5") == "44"
    assert calc("2^10") == "1024"
    assert calc("(25 * 48) + sqrt(625)") == "1225"


def test_calculator_errors_are_safe():
    assert "division by zero" in calc("10/0")
    assert calc("__import__('os').system('echo hacked')").startswith("Error")
    assert calc("open('x')").startswith("Error")


def test_file_reader_lists_and_reads():
    assert "notes.txt" in read("list")
    assert "Budget" in read("notes.txt")


def test_file_reader_blocks_path_traversal():
    assert "access denied" in read("../agent.py")
    assert "access denied" in read("../../README.md")
    assert "not found" in read("missing.txt")
