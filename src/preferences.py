def get_preferences():

    features = [
        "affordability",
        "career",
        "business",
        "cafes",
        "fitness",
        "nightlife",
        "nature",
        "climate",
        "mobility",
        "safety",
        "social"
    ]

    user_preferences = {}

    print("\n========================================")
    print("      YOUR CITY PREFERENCES")
    print("========================================")
    print("\nRate each factor from 1 to 10.")
    print("1 = Not important")
    print("10 = Extremely important\n")

    for feature in features:

        while True:

            try:
                value = float(
                    input(
                        f"How important is {feature}? (1-10): "
                    )
                )

                if 1 <= value <= 10:
                    user_preferences[feature] = value
                    break

                print("Please enter a number between 1 and 10.")

            except ValueError:
                print("Please enter a valid number.")

    return user_preferences