def human_readable_number(num: int | float) -> str:
    suffixes = [
        (1_000_000_000_000, "Trillion"),
        (1_000_000_000, "Billion"),
        (1_000_000, "Million"),
        (1_000, "Thousand"),
    ]

    for value, name in suffixes:
        if abs(num) >= value:
            return f"{num / value:.2f} {name}"

    return str(num)