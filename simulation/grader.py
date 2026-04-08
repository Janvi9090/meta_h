LOW, HIGH = 10, 50

def grade(concentrations):
    if len(concentrations) == 0:
        return 0.0
    good = sum(1 for c in concentrations if LOW <= c <= HIGH)
    return good / len(concentrations)