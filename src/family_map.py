from collections import defaultdict

# Exact scenario name for each bag_id
BAG_TO_NAME = {
    0: "Empty, robot vel 0.3",
    1: "Empty, robot vel 0.5",
    2: "1 MIR, MIR vel 0.8, robot vel 0.3",
    3: "1 MIR, MIR vel 0.8, robot vel 0.5",
    4: "1 MIR, MIR vel 1.2, robot vel 0.3",
    5: "1 MIR, MIR vel 1.2, robot vel 0.5",
    6: "1 person, track 1, robot vel 0.3",
    7: "1 person, track 1, robot vel 0.5",
    8: "2 persons, track 1, robot vel 0.3",
    9: "2 persons, track 1, robot vel 0.5",
    10: "1 person, track 2, robot vel 0.5",
    11: "2 robots, main robot vel 0.3",
    12: "2 robots, main robot vel 0.5",
}

# Scenario family for grouped cross-validation
BAG_TO_FAMILY = {
    0: "Empty",
    1: "Empty",
    2: "1 MIR, MIR vel 0.8",
    3: "1 MIR, MIR vel 0.8",
    4: "1 MIR, MIR vel 1.2",
    5: "1 MIR, MIR vel 1.2",
    6: "1 person, track 1",
    7: "1 person, track 1",
    8: "2 persons, track 1",
    9: "2 persons, track 1",
    10: "1 person, track 2",
    11: "2 robots",
    12: "2 robots",
}


def build_family_to_bags():
    family_to_bags = defaultdict(list)
    for bag_id, family in BAG_TO_FAMILY.items():
        family_to_bags[family].append(bag_id)

    for family in family_to_bags:
        family_to_bags[family] = sorted(family_to_bags[family])

    return {family: family_to_bags[family] for family in sorted(family_to_bags.keys())}


FAMILY_TO_BAGS = build_family_to_bags()
FAMILY_NAMES = list(FAMILY_TO_BAGS.keys())


def get_bag_name(bag_id: int) -> str:
    return BAG_TO_NAME[bag_id]


def get_family_name(bag_id: int) -> str:
    return BAG_TO_FAMILY[bag_id]


def get_bags_in_family(family_name: str):
    return FAMILY_TO_BAGS[family_name]


def validate_family_map():
    bag_ids_name = set(BAG_TO_NAME.keys())
    bag_ids_family = set(BAG_TO_FAMILY.keys())

    if bag_ids_name != bag_ids_family:
        missing_in_name = sorted(bag_ids_family - bag_ids_name)
        missing_in_family = sorted(bag_ids_name - bag_ids_family)
        raise ValueError(
            "BAG_TO_NAME and BAG_TO_FAMILY do not cover the same bag_ids. "
            f"Missing in BAG_TO_NAME: {missing_in_name}. "
            f"Missing in BAG_TO_FAMILY: {missing_in_family}."
        )

    if len(FAMILY_TO_BAGS) == 0:
        raise ValueError("No families were created.")

    total_bags = sum(len(v) for v in FAMILY_TO_BAGS.values())
    if total_bags != len(BAG_TO_NAME):
        raise ValueError(
            f"Family mapping inconsistent: total bags in families = {total_bags}, "
            f"but BAG_TO_NAME has {len(BAG_TO_NAME)} entries."
        )


validate_family_map()