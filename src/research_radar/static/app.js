const storageKey = "research-radar:saved-papers";

function savedPapers() {
  try {
    return new Set(JSON.parse(localStorage.getItem(storageKey) || "[]"));
  } catch (error) {
    console.warn("Unable to read saved papers", error);
    return new Set();
  }
}

function renderSavedState() {
  const saved = savedPapers();
  document.querySelectorAll("[data-save]").forEach((button) => {
    const isSaved = saved.has(button.dataset.save);
    button.classList.toggle("saved", isSaved);
    button.textContent = isSaved ? "★" : "☆";
    button.setAttribute("aria-label", isSaved ? "Remove saved paper" : "Save paper");
  });

  const savedOnly = document.querySelector("#saved-only")?.checked;
  document.querySelectorAll("[data-paper-id]").forEach((paper) => {
    paper.hidden = Boolean(savedOnly && !saved.has(paper.dataset.paperId));
  });
}

document.addEventListener("click", (event) => {
  const button = event.target.closest("[data-save]");
  if (!button) return;

  const saved = savedPapers();
  if (saved.has(button.dataset.save)) saved.delete(button.dataset.save);
  else saved.add(button.dataset.save);
  localStorage.setItem(storageKey, JSON.stringify([...saved]));
  renderSavedState();
});

document.querySelector("#saved-only")?.addEventListener("change", renderSavedState);
renderSavedState();
