import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import { Roboto, Inter, Montserrat } from "next/font/google";
const roboto = Roboto({
  weight: "400",
  subsets: ["latin"],
});
const inter = Inter({ subsets: ["latin"] });
const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});
const mostserrat = Montserrat({
  weight: "400",
  subsets: ["latin"],
});
const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata = {
  title: "Curiosity",
  description: "Curiosity is a chatbot that helps you learn new things.",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body className={mostserrat.className}>{children}</body>
    </html>
  );
}
