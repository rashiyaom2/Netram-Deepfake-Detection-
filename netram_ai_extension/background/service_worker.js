/**
 * Netram AI Deepfake Shield - Background Service Worker (Manifest V3)
 * Manages extension state, first-run onboarding launch, 5-tier threat badge icons, and notifications.
 */

chrome.runtime.onInstalled.addListener((details) => {
  chrome.storage.local.set({
    overlayEnabled: true,
    audioAlertEnabled: true,
    strictness: "balanced",
  });
  console.log("[Netram AI] Shield initialized with default enterprise settings.");

  // Automatically launch the interactive onboarding experience on first install/update
  chrome.tabs.create({ url: chrome.runtime.getURL("onboarding/onboarding.html") });
});

// Update badge icon when deepfakes/threat levels are detected
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.type === "UPDATE_BADGE") {
    const level = (request.threat_level || request.status || "").toUpperCase();
    if (level === "CRITICAL" || level === "DEEPFAKE") {
      chrome.action.setBadgeText({ text: "!" });
      chrome.action.setBadgeBackgroundColor({ color: "#f06060" });
    } else if (level === "HIGH") {
      chrome.action.setBadgeText({ text: "!" });
      chrome.action.setBadgeBackgroundColor({ color: "#ff9f43" });
    } else if (level === "MODERATE" || level === "REVIEW") {
      chrome.action.setBadgeText({ text: "?" });
      chrome.action.setBadgeBackgroundColor({ color: "#f5a623" });
    } else if (level === "LOW") {
      chrome.action.setBadgeText({ text: "·" });
      chrome.action.setBadgeBackgroundColor({ color: "#4fc3f7" });
    } else {
      chrome.action.setBadgeText({ text: "" });
    }
  }
  sendResponse({ ok: true });
});
