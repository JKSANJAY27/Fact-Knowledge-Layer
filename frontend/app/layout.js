import './globals.css';

export const metadata = {
  title: 'Fact Knowledge Layer - Institutional PDF Intelligence',
  description: 'Atomic fact extraction, canonical normalization, deterministic reconciliation, and grounded retrieval.',
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
