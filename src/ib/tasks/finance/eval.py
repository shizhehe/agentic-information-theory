import re 
import math 

def check_answer_type(s):
    # Check for a percentage
    if re.fullmatch(r"\d+(\.\d+)?%", s):
        return "percentage"
    # Check for a dollar amount
    elif re.fullmatch(r"\$\d+(\.\d+)?", s):
        return "dollar"
    # Check for a plain number
    elif re.fullmatch(r"\d+(\.\d+)?", s):
        return "number"
    else:
        return "string"

def extract_answer_from_sample(sample: str):
    sample = sample.replace("<|eot_id|>", "").strip(". \n")
    match = re.search(r"Final answer: (.*)", sample)
    if match:
        answer = match.group(1)
        answer = answer.strip(".")
        return answer
    else:
        return sample


def is_correct_finance(pred: str, answer: str, log_error: bool = True):
    """
    Check if the prediction is correct, given the ground truth answer.
    """
    REMOVE_WORDS = ["usd", "dollars", "people", "approximately", "million", "billion"]
    try:
        # first lower-case pred (answer should already be numeric+symbol)
        pred = pred.lower()
        for word in REMOVE_WORDS:
            pred = pred.replace(word, "")
        pred = pred.strip(". \n")
        answer = answer.lower()
        for word in REMOVE_WORDS:
            answer = answer.replace(word, "")
        answer = answer.replace("people", "")
        answer = answer.strip(". \n")
        if pred == answer:
            return True

        multiplier = 1
        # if "million" in pred:
        #     pred = pred.replace("million", "")
        #     multiplier = 1e6
        # elif "billion" in pred:
        #     pred = pred.replace("billion", "")
        #     multiplier = 1e9
        # else:
        #     multiplier = 1
        pred = pred.replace(",", "")
        answer = answer.replace(",", "")
        # if answer is dollar and pred is not percentage (make sure we are not comparing percentage to dollar)
        if answer.startswith("$") and not pred.endswith("%"): 
            # remove dollar sign
            pred = pred.replace("$", "")
            answer = answer.replace("$", "")
            if (float(pred) < 1.0) and (float(answer) < 1.0):
                # if both are less than 1, be more strict with absolute tolerance
                abs_tol = 0.1
            else:
                abs_tol = 0.9
            return (
                math.isclose(float(pred), float(answer), rel_tol=1e-5, abs_tol=abs_tol) or 
                math.isclose(float(pred) * multiplier, float(answer), rel_tol=1e-5, abs_tol=abs_tol)
            )
        elif answer.endswith("%") and not pred.startswith("$"):
            # percentage deemed correct if within 0.99% of the answer
            # remove percentage sign
            pred = pred.replace("%", "")
            answer = answer.replace("%", "")
            return math.isclose(float(pred), float(answer), abs_tol=0.99)
    except ValueError as e:
        if pred == answer:
            return True
        if log_error:
            print(f"Error processing {pred} and {answer}: {e}")
        return False

    return pred == answer