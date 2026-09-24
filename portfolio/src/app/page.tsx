import Background from "@/components/Background";
import Nav from "@/components/Nav";
import Hero from "@/components/Hero";
import Work from "@/components/Work";
import Services from "@/components/Services";
import About from "@/components/About";
import Contact from "@/components/Contact";
import Footer from "@/components/Footer";
import Handoff from "@/components/Handoff";

export default function Home() {
  return (
    <>
      <Background />
      <Nav />
      <main>
        <Hero />
        <Work />
        <Services />
        <About />
        <Contact />
      </main>
      <Footer />
      <Handoff />
    </>
  );
}
