/**
 * Netram AI — Legal Trust Center Controller
 * Manages terms acceptance state, persistence, print action, and active TOC scroll observer.
 */
document.addEventListener("DOMContentLoaded", () => {
  const cb = document.getElementById("cb-accept-terms");
  const btn = document.getElementById("btn-accept-terms");
  const printBtn = document.getElementById("btn-print");

  if (printBtn) {
    printBtn.addEventListener("click", () => window.print());
  }

  function updateAcceptState() {
    if (btn && cb) {
      btn.disabled = !cb.checked;
    }
  }

  if (cb) {
    cb.addEventListener("change", updateAcceptState);
  }

  function completeAcceptance() {
    const timestamp = new Date().toISOString();
    try {
      if (typeof chrome !== "undefined" && chrome.storage && chrome.storage.local) {
        chrome.storage.local.set({ termsAccepted: true, termsAcceptedTimestamp: timestamp }, () => {
          finishUI();
        });
      } else {
        localStorage.setItem("netram_terms_accepted", "true");
        finishUI();
      }
    } catch (e) {
      finishUI();
    }
  }

  function finishUI() {
    if (btn) {
      btn.textContent = "Accepted ✓";
      btn.disabled = true;
    }
  }

  if (btn) {
    btn.addEventListener("click", completeAcceptance);
  }

  // Check if already accepted
  if (typeof chrome !== "undefined" && chrome.storage && chrome.storage.local) {
    chrome.storage.local.get(["termsAccepted"], (res) => {
      if (res.termsAccepted) {
        if (cb) cb.checked = true;
        finishUI();
      }
    });
  } else if (localStorage.getItem("netram_terms_accepted") === "true") {
    if (cb) cb.checked = true;
    finishUI();
  }

  // Active TOC link on scroll
  const sections = document.querySelectorAll(".legal-section");
  const links = document.querySelectorAll(".toc-links a");
  if (sections.length && links.length && "IntersectionObserver" in window) {
    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          links.forEach((l) => l.classList.remove("active"));
          const match = document.querySelector(`.toc-links a[href="#${entry.target.id}"]`);
          if (match) match.classList.add("active");
        }
      });
    }, { rootMargin: "-40% 0px -50% 0px" });
    sections.forEach((s) => observer.observe(s));
  }
});
