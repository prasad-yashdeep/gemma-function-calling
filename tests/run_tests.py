import unittest
import os
import sys

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import test modules
from tests.test_api_converter_and_sdk_definition import (
    TestAPIConverter,
    TestFunctionDefinition,
    TestFunctionRegistry,
    TestFunctionCallResult
)
from tests.test_runtime import (
    TestGemmaRuntime,
    TestReActExecutor,
    TestConversation,
    TestConversationManager,
    TestImageProcessor
)

if __name__ == "__main__":
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add tests from test_api_converter_and_sdk_definition
    test_suite.addTest(unittest.makeSuite(TestAPIConverter))
    test_suite.addTest(unittest.makeSuite(TestFunctionDefinition))
    test_suite.addTest(unittest.makeSuite(TestFunctionRegistry))
    test_suite.addTest(unittest.makeSuite(TestFunctionCallResult))
    
    # Add tests from test_runtime
    test_suite.addTest(unittest.makeSuite(TestGemmaRuntime))
    test_suite.addTest(unittest.makeSuite(TestReActExecutor))
    test_suite.addTest(unittest.makeSuite(TestConversation))
    test_suite.addTest(unittest.makeSuite(TestConversationManager))
    test_suite.addTest(unittest.makeSuite(TestImageProcessor))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(test_suite)
