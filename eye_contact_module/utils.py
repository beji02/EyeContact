
def time_string_to_ms(time_str: str) -> float:
    minutes, seconds = map(int, time_str.split(':'))
    return (minutes * 60 + seconds) * 1000