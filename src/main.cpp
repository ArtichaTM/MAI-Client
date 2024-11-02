#include <string>
#include <iostream>

#include <SFML/Graphics.hpp>

#include "elements/interactive/button.hpp"
#include "elements/tabs/system.hpp"
#include "config.hpp"


TabSystem* build_ui() {
    TabSystem* tabsys = new TabSystem(10.f, sf::Color::Green);
    Tab* tab1 = tabsys->addTab("Overview");
    Tab* tab2 = tabsys->addTab("AI modules influences");
    return tabsys;
}

int main() {
    sf::RenderWindow window(
        sf::VideoMode(800, 600),
        MAICLient_NAME,
        sf::Style::Titlebar |
        sf::Style::Close
    );
    ROOT_WINDOW = &window;
    TabSystem* tabsys = build_ui();
    window.setFramerateLimit(6);

    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed) {
                window.close();
                break;
            }
            tabsys->handleEvent(event);
        }

        window.clear();
        window.draw(*tabsys);
        window.display();
    }

    delete tabsys;
    return 0;
}
