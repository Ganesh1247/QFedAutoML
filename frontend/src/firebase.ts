import { initializeApp } from "firebase/app";

// Web app's Firebase configuration
export const firebaseConfig = {
  apiKey: "AIzaSyDkKxHuNRmXHeB1ieBLHW3NiDwakv0CY14",
  authDomain: "qfedauto.firebaseapp.com",
  projectId: "qfedauto",
  storageBucket: "qfedauto.firebasestorage.app",
  messagingSenderId: "211318388642",
  appId: "1:211318388642:web:69e277910dfc2ebf3364fb"
};

// Initialize Firebase
export const app = initializeApp(firebaseConfig);
