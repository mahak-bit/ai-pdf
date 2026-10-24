import type { Metadata, Viewport } from "next";
import { Instrument_Sans, Source_Serif_4 } from "next/font/google";
import "./globals.css";

const instrument = Instrument_Sans({
  variable: "--font-instrument",
  subsets: ["latin"],
});

const sourceSerif = Source_Serif_4({
  variable: "--font-source-serif",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "DocuMind AI",
  description: "Ask questions about a PDF and get answers cited to the exact page.",
};

export const viewport: Viewport = {
  themeColor: "#faf8f2",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className={`${instrument.variable} ${sourceSerif.variable} h-full`}>
      <body className="h-full">{children}</body>
    </html>
  );
}
