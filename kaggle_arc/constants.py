import matplotlib
import seaborn as sns

DATADIR = "../input/abstraction-and-reasoning-challenge"

WORKDIR = "../working"
TEST_SAVEPATH = "../working/submission.csv"

PALETTE = sns.crayon_palette(
    (
        "Eggplant,Aquamarine,Jungle Green,Atomic Tangerine,Blue Bell,Wisteria,"
        + "Banana Mania,Blue Violet,Carnation Pink,Cerise"
    ).split(",")
)  # list(sns.crayons)[:10])
COLORMAP = matplotlib.colors.ListedColormap(PALETTE)
NORM = matplotlib.colors.Normalize(vmin=0, vmax=9)
