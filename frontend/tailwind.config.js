/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        chatBg: "#212121",
        userMsg: "#2f2f2f",
        sidebarBg: "#171717",
        inputBg: "#2f2f2f",
        borderLight: "#424242"
      }
    },
  },
  plugins: [],
}
