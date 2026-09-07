"""The token and mass contract of the fixed CUDA PMC kernel."""

PMC_RESIDUES = {
    "G": 57.021464,
    "A": 71.037114,
    "S": 87.032028,
    "P": 97.052764,
    "V": 99.068414,
    "T": 101.047670,
    "C+57.021": 160.030649,
    "L": 113.084064,
    "I": 113.084064,
    "N": 114.042927,
    "D": 115.026943,
    "Q": 128.058578,
    "K": 128.094963,
    "E": 129.042593,
    "M": 131.040485,
    "H": 137.058912,
    "F": 147.068414,
    "R": 156.101111,
    "Y": 163.063329,
    "W": 186.079313,
    "M+15.995": 147.035400,
    "N+0.984": 115.026943,
    "Q+0.984": 129.042594,
    "+42.011": 42.010565,
    "+43.006": 43.005814,
    "-17.027": -17.026549,
    "+43.006-17.027": 25.980265,
}


def validate_pmc_schema(decoder, time_steps):
    symbols = [decoder._idx2aa[index] for index in range(len(decoder._idx2aa))]
    if time_steps != 40 or symbols != [*PMC_RESIDUES, "_"]:
        raise ValueError("PMC requires the frozen 28-token order and max_length=40.")
    if any(
        abs(float(decoder._peptide_mass.masses[key]) - mass) > 1e-6
        for key, mass in PMC_RESIDUES.items()
    ):
        raise ValueError("PMC residue masses differ from the frozen CUDA schema.")
