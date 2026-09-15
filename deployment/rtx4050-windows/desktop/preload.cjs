"use strict";
const { contextBridge, ipcRenderer } = require("electron");
contextBridge.exposeInMainWorld("visioncortexDesktop", {
  getState: () => ipcRenderer.invoke("desktop:get-state"),
  scanStorage: () => ipcRenderer.invoke("desktop:scan-storage"),
  chooseStorage: () => ipcRenderer.invoke("desktop:choose-storage"),
  chooseSource: () => ipcRenderer.invoke("desktop:choose-source"),
  saveStorage: (value) => ipcRenderer.invoke("desktop:save-storage",value),
  continue: () => ipcRenderer.invoke("desktop:continue"),
  saveConnection: (value) => ipcRenderer.invoke("desktop:save-connection", value),
  retry: () => ipcRenderer.invoke("desktop:retry"),
  openLogs: () => ipcRenderer.invoke("desktop:open-logs"),
  openModelDoc: (provider,model) => ipcRenderer.invoke("desktop:open-model-doc",provider,model),
  onState: (callback) => {
    const listener = (_event, state) => callback(state);
    ipcRenderer.on("desktop:state", listener);
    return () => ipcRenderer.removeListener("desktop:state", listener);
  },
});
