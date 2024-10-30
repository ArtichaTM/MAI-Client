#include <SFML/Graphics.hpp>

#include "elements/button.hpp"

int main() {
    sf::RenderWindow window(sf::VideoMode(800, 600), "Simple Button Example");
    Button button(300, 250, 200, 50, "Click Me");
    window.setVerticalSyncEnabled(false);

    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed)
                window.close();
    //         if (event.type == sf::Event::MouseButtonPressed && event.mouseButton.button == sf::Mouse::Left) {
    //             if (button.isMouseOver(sf::Mouse::getPosition(window))) {
    //                 button.onClick();
    //             }
            // }
        }

    //     button.update(sf::Mouse::getPosition(window));

        window.clear();
    //     button.draw(window);
        window.display();
    }

    return 0;
}