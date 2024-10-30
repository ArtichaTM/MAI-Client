#include <string>

#include <SFML/Graphics.hpp>

#include "elements/button.hpp"
#include "elements/tabs/system.hpp"
#include "config.hpp"


TabSystem build_ui() {
    TabSystem tabs(10.f);
    return tabs;
}


int main() {
    sf::RenderWindow window(
        sf::VideoMode(800, 600),
        MAICLient_NAME,
        sf::Style::Close
    );
    ROOT_WINDOW = &window;
    TabSystem sys = build_ui();

    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed) {
                window.close();
                break;
            }
        }

        window.clear();
        sys.draw(window);
        window.display();
    }

    return 0;
}
