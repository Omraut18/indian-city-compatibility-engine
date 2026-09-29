def calculate_score(city, user_preferences):

    total_score = 0
    total_weight = 0

    contributions = {}

    for feature in user_preferences:

        preference = user_preferences[feature]
        city_score = city[feature]

        # How well the city satisfies the user's preference
        match_score = (
            10 - abs(preference - city_score)
        )

        # Important preferences have more influence
        contribution = (
            preference * match_score
        )

        total_score += contribution
        total_weight += preference

        contributions[feature] = contribution

    score = total_score / total_weight

    return score, contributions