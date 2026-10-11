import { useEffect, useState, type MouseEvent, type ReactNode } from "react";

// A two-page app doesn't need a routing library. The browser's History API
// (pushState + the popstate event) is enough to change the URL and react to Back/Forward.

const ROUTE_EVENT = "glassbox:navigate";

export function navigate(path: string) {
  if (path === window.location.pathname) return;
  window.history.pushState(null, "", path);
  window.dispatchEvent(new Event(ROUTE_EVENT));
  window.scrollTo({ top: 0 });
}

export function usePath(): string {
  const [path, setPath] = useState(window.location.pathname);
  useEffect(() => {
    const update = () => setPath(window.location.pathname);
    window.addEventListener("popstate", update); // Back / Forward buttons
    window.addEventListener(ROUTE_EVENT, update); // our own navigate()
    return () => {
      window.removeEventListener("popstate", update);
      window.removeEventListener(ROUTE_EVENT, update);
    };
  }, []);
  return path;
}

/** A normal <a> (so right-click and "open in new tab" work) that navigates without a reload. */
export function Link(props: { to: string; className?: string; children: ReactNode }) {
  const onClick = (event: MouseEvent) => {
    if (event.metaKey || event.ctrlKey || event.shiftKey) return; // let the browser open a new tab
    event.preventDefault();
    navigate(props.to);
  };
  return (
    <a href={props.to} className={props.className} onClick={onClick}>
      {props.children}
    </a>
  );
}
