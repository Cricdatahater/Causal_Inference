"""Execute Stage 6 and persist its notebook outputs, figures, and result tables."""
from pathlib import Path
import nbformat
from nbclient import NotebookClient


def main():
    root = Path(__file__).resolve().parents[1]
    path = root / "notebooks" / "04_doubly_robust_policy_evaluation.ipynb"
    notebook = nbformat.read(path, as_version=4)
    NotebookClient(notebook, timeout=600, kernel_name="python3",
                   resources={"metadata": {"path": str(root)}}).execute()
    nbformat.write(notebook, path)
    for output in notebook.cells[-1].get("outputs", []):
        if output.output_type == "stream":
            print(output.text)
    print(f"Saved executed notebook: {path}")


if __name__ == "__main__":
    main()
