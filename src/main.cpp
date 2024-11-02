#include <string>
#include <iostream>

#include <SFML/Graphics.hpp>

#include "./builders/header.hpp"
#include "config.hpp"

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
