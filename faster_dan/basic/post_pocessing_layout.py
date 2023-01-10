import numpy as np
from faster_dan.Datasets.dataset_formatters.read2016_formatter import SEM_TOKENS as READ_TOKENS
from faster_dan.Datasets.dataset_formatters.rimes_formatter import SEM_TOKENS as RIMES_TOKENS


class PostProcessingModule:
    """
    Forward pass post processing
    Add/remove layout tokens only to:
     - respect token hierarchy
     - complete/remove unpaired tokens
    """

    def __init__(self):
        self.prediction = None
        self.confidence = None
        self.num_op = 0

    def post_processing(self):
        raise NotImplementedError

    def post_process(self, prediction, confidence_score=None):
        """
        Apply dataset-specific post-processing
        """
        self.num_op = 0
        self.prediction = list(prediction)
        self.confidence = list(confidence_score) if confidence_score is not None else None
        if self.confidence is not None:
            assert len(self.prediction) == len(self.confidence)
        return self.post_processing()

    def insert_label(self, index, label):
        """
        Insert token at specific index. The associated confidence score is set to 0.
        """
        self.prediction.insert(index, label)
        if self.confidence is not None:
            self.confidence.insert(index, 0)
        self.num_op += 1

    def del_label(self, index):
        """
        Remove the token at a specific index.
        """
        del self.prediction[index]
        if self.confidence is not None:
            del self.confidence[index]
        self.num_op += 1


class PostProcessingModuleREAD(PostProcessingModule):
    """
    Specific post-processing for the READ 2016 dataset at single-page and double-page levels
    """
    def __init__(self):
        super(PostProcessingModuleREAD, self).__init__()

        self.tokens = READ_TOKENS

    def post_processing_page_labels(self):
        """
        Correct tokens of page detection.
        """
        ind = 0
        while ind != len(self.prediction):
            # Label must start with a begin-page token
            if ind == 0 and self.prediction[ind] != self.tokens["page"]:
                self.insert_label(0, self.tokens["page"])
                continue
            # There cannot be tokens out of begin-page end-page scope: begin-page must be preceded by end-page
            if self.prediction[ind] == self.tokens["page"] and ind != 0 and self.prediction[ind - 1] != self.tokens["page"].upper():
                self.insert_label(ind, self.tokens["page"].upper())
                continue
            # There cannot be tokens out of begin-page end-page scope: end-page must be followed by begin-page
            if self.prediction[ind] == self.tokens["page"].upper() and ind < len(self.prediction) - 1 and self.prediction[ind + 1] != self.tokens["page"]:
                self.insert_label(ind + 1, self.tokens["page"])
            ind += 1
        # Label must start with a begin-page token even for empty prediction
        if len(self.prediction) == 0:
            self.insert_label(0, self.tokens["page"])
            ind += 1
        # Label must end with a end-page token
        if self.prediction[-1] != self.tokens["page"].upper():
            self.insert_label(ind, self.tokens["page"].upper())

    def post_processing(self):
        """
        Correct tokens of page number, section, body and annotations.
        """
        self.post_processing_page_labels()
        ind = 0
        begin_token = None
        in_section = False
        while ind != len(self.prediction):
            # each tags must be closed while changing page
            if self.prediction[ind] == self.tokens["page"].upper():
                if begin_token is not None:
                    self.insert_label(ind, begin_token.upper())
                    begin_token = None
                    ind += 1
                elif in_section:
                    self.insert_label(ind, self.tokens["section"].upper())
                    in_section = False
                    ind += 1
                else:
                    ind += 1
                continue
            # End token is removed if the previous begin token does not match with it
            if self.prediction[ind] in [self.tokens["number"].upper(), self.tokens["annotation"].upper(), self.tokens["body"].upper()]:
                if begin_token == self.prediction[ind].lower():
                    begin_token = None
                    ind += 1
                else:
                    self.del_label(ind)
                continue
            if self.prediction[ind] == self.tokens["section"].upper():
                # each sub-tags must be closed while closing section
                if in_section:
                    if begin_token is None:
                        ind += 1
                    else:
                        self.insert_label(ind, begin_token.upper())
                        begin_token = None
                        ind += 2
                    in_section = False
                else:
                    self.del_label(ind)
                continue
            if self.prediction[ind] == self.tokens["section"]:
                # A sub-tag must be closed before opening a section
                if begin_token is not None:
                    self.insert_label(ind, begin_token.upper())
                    begin_token = None
                    ind += 1
                # A section must be closed before opening a new one
                elif in_section:
                    self.insert_label(ind, self.tokens["section"].upper())
                    in_section = False
                    ind += 1
                else:
                    in_section = True
                    ind += 1
                continue
            if self.prediction[ind] == self.tokens["number"]:
                # Page number cannot be in section: a started section must be closed
                if begin_token is None:
                    if in_section:
                        in_section = False
                        self.insert_label(ind, self.tokens["section"].upper())
                        ind += 1
                    begin_token = self.prediction[ind]
                    ind += 1
                else:
                    self.insert_label(ind, begin_token.upper())
                    begin_token = None
                    ind += 1
                continue
            if self.prediction[ind] in [self.tokens["annotation"], self.tokens["body"]]:
                # Annotation and body must be in section
                if begin_token is None:
                    if in_section:
                        begin_token = self.prediction[ind]
                        ind += 1
                    else:
                        in_section = True
                        self.insert_label(ind, self.tokens["section"])
                        ind += 1
                # Previous sub-tag must be closed
                else:
                    self.insert_label(ind, begin_token.upper())
                    begin_token = None
                    ind += 1
                continue
            ind += 1
        res = "".join(self.prediction)
        if self.confidence is not None:
            return res, np.array(self.confidence)
        return res


class PostProcessingModuleRIMES(PostProcessingModule):
    """
    Specific post-processing for the RIMES dataset at page level
    """
    def __init__(self):
        super(PostProcessingModuleRIMES, self).__init__()
        self.tokens = RIMES_TOKENS
        self.end_tokens = [b.upper() for b in self.tokens.values()]

    def post_processing(self):
        ind = 0
        begin_token = None
        while ind != len(self.prediction):
            char = self.prediction[ind]
            # a tag must be closed before starting a new one
            if char in self.tokens.values():
                if begin_token is None:
                    ind += 1
                else:
                    self.insert_label(ind, begin_token.upper())
                    ind += 2
                begin_token = char
                continue
            # an end token without prior corresponding begin token is removed
            elif char in self.end_tokens:
                if begin_token == char.lower():
                    ind += 1
                    begin_token = None
                else:
                    self.del_label(ind)
                continue
            else:
                ind += 1
        # a tag must be closed
        if begin_token is not None:
            self.insert_label(ind+1, begin_token.upper())
        res = "".join(self.prediction)
        if self.confidence is not None:
            return res, np.array(self.confidence)
        return res
