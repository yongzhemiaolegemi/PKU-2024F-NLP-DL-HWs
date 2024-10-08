# An autograder for Python homework
import importlib.util
import traceback

exercises = {
    'flatten_list': {
        'function_name': 'flatten_list',
        'test_cases': [
            ([[1, 2, 3], [4, 5], [6]], [1, 2, 3, 4, 5, 6]),
            ([[1, 2], [3, 4]], [1, 2, 3, 4]),
            ([], []),
        ]
    },
    'char_count': {
        'function_name': 'char_count',
        'test_cases': [
            ("hello world", {'h': 1, 'e': 1, 'l': 3, 'o': 2, ' ': 1, 'w': 1, 'r': 1, 'd': 1}),
            ("aaa", {'a': 3}),
            ("", {})
        ]
    }
}

def load_module(file_path):
    """Dynamically load a module from a given file path."""
    spec = importlib.util.spec_from_file_location("submission", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def run_tests(module, exercises):
    """Run tests for all exercises and print results."""
    results = {}
    
    for exercise_name, details in exercises.items():
        function_name = details['function_name']
        test_cases = details['test_cases']
        results[exercise_name] = {'passed': 0, 'failed': 0, 'errors': []}
        
        if not hasattr(module, function_name):
            results[exercise_name]['errors'].append(f"Function '{function_name}' not found.")
            continue
        
        function = getattr(module, function_name)
        
        for i, (input_data, expected_output) in enumerate(test_cases):
            try:
                # If input_data is a tuple, unpack it; otherwise, pass it as a single argument.
                if isinstance(input_data, tuple):
                    result = function(*input_data)
                else:
                    result = function(input_data)
                
                if result == expected_output:
                    results[exercise_name]['passed'] += 1
                else:
                    results[exercise_name]['failed'] += 1
                    results[exercise_name]['errors'].append(
                        f"Test case {i + 1} failed: input={input_data}, expected={expected_output}, got={result}"
                    )
            except Exception as e:
                results[exercise_name]['failed'] += 1
                results[exercise_name]['errors'].append(
                    f"Test case {i + 1} raised an exception: {traceback.format_exc()}"
                )
    
    return results

def print_results(results):
    """Print the results in a readable format."""
    for exercise_name, result in results.items():
        print(f"Exercise: {exercise_name}")
        print(f"  Passed: {result['passed']}")
        print(f"  Failed: {result['failed']}")
        if result['errors']:
            print("  Errors:")
            for error in result['errors']:
                print(f"    - {error}")
        print("\n")

if __name__ == "__main__":
    # Replace 'submission.py' with the path to the user's submission file.
    submission_file = 'submission.py'
    user_module = load_module(submission_file)
    results = run_tests(user_module, exercises)
    print_results(results)