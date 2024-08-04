from validex.base import DataCleaner


class TestDataCleaner:
    # clean text with multiple spaces to single space
    def test_clean_text_with_multiple_spaces(self):
        cleaner = DataCleaner()
        input_text = "This  is   a    test"
        expected_output = "This is a test"
        assert cleaner.clean(input_text) == expected_output

    # clean text that is already clean
    def test_clean_text_already_clean(self):
        cleaner = DataCleaner()
        input_text = "This is a test"
        expected_output = "This is a test"
        assert cleaner.clean(input_text) == expected_output
