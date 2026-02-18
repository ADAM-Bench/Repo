const revealElements = Array.from(document.querySelectorAll(".reveal"));
const navLinks = Array.from(document.querySelectorAll("#mainNav a"));
const statNumbers = Array.from(document.querySelectorAll("[data-count]"));
const yearNode = document.getElementById("year");

if (yearNode) {
  yearNode.textContent = String(new Date().getFullYear());
}

if ("IntersectionObserver" in window) {
  const revealObserver = new IntersectionObserver(
    (entries, observer) => {
      for (const entry of entries) {
        if (!entry.isIntersecting) {
          continue;
        }
        entry.target.classList.add("is-visible");
        observer.unobserve(entry.target);
      }
    },
    { threshold: 0.18 }
  );

  for (const node of revealElements) {
    revealObserver.observe(node);
  }
} else {
  for (const node of revealElements) {
    node.classList.add("is-visible");
  }
}

const formatInt = new Intl.NumberFormat("en-US");

function animateCount(node) {
  const target = Number(node.dataset.count || "0");
  if (!Number.isFinite(target)) {
    return;
  }
  const duration = 900;
  const start = performance.now();

  function frame(now) {
    const progress = Math.min((now - start) / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3);
    node.textContent = formatInt.format(Math.round(target * eased));
    if (progress < 1) {
      requestAnimationFrame(frame);
    }
  }

  requestAnimationFrame(frame);
}

if ("IntersectionObserver" in window) {
  const countObserver = new IntersectionObserver(
    (entries, observer) => {
      for (const entry of entries) {
        if (!entry.isIntersecting) {
          continue;
        }
        animateCount(entry.target);
        observer.unobserve(entry.target);
      }
    },
    { threshold: 0.5 }
  );

  for (const node of statNumbers) {
    countObserver.observe(node);
  }
} else {
  for (const node of statNumbers) {
    animateCount(node);
  }
}

function syncActiveSection() {
  const current = navLinks
    .map((link) => {
      const id = link.getAttribute("href");
      if (!id || !id.startsWith("#")) {
        return null;
      }
      const node = document.querySelector(id);
      if (!node) {
        return null;
      }
      return { link, top: node.getBoundingClientRect().top };
    })
    .filter(Boolean)
    .sort((a, b) => Math.abs(a.top) - Math.abs(b.top))[0];

  for (const link of navLinks) {
    link.classList.remove("is-active");
  }
  if (current && current.link) {
    current.link.classList.add("is-active");
  }
}

let ticking = false;
window.addEventListener(
  "scroll",
  () => {
    if (ticking) {
      return;
    }
    ticking = true;
    requestAnimationFrame(() => {
      syncActiveSection();
      ticking = false;
    });
  },
  { passive: true }
);
syncActiveSection();

const copyBtn = document.getElementById("copyBib");
const bibNode = document.querySelector("#bibtex code");
if (copyBtn && bibNode) {
  copyBtn.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(bibNode.textContent.trim());
      copyBtn.textContent = "Copied";
      window.setTimeout(() => {
        copyBtn.textContent = "Copy BibTeX";
      }, 1200);
    } catch {
      copyBtn.textContent = "Copy Failed";
      window.setTimeout(() => {
        copyBtn.textContent = "Copy BibTeX";
      }, 1200);
    }
  });
}
