// The theme lives on <html data-theme="light|dark">. A tiny script in <head> sets it before first paint
// (saved choice, else the system setting), so there is no flash of the wrong theme.
export const THEME_SCRIPT = `(function(){var d=document.documentElement,m=window.matchMedia("(prefers-color-scheme: dark)");function s(){var t=null;try{t=localStorage.getItem("theme")}catch(e){}d.dataset.theme=t==="dark"||t==="light"?t:m.matches?"dark":"light"}s();m.addEventListener("change",s)})()`;
