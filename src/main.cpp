#include <string>

#include <SFML/Graphics.hpp>

#include "elements/button.hpp"
#include "elements/tabs/system.hpp"
#include "config.hpp"


TabSystem* build_ui() {
    TabSystem* tabsys = new TabSystem(10.f, sf::Color::Green);
    Tab* tab1 = tabsys->addTab("Overview");
    Tab* tab2 = tabsys->addTab("AI modules influences");
    tab1->AddElement(new Button(
        10, tabsys->getHeight()+10, "Run",
        PATH_FONT_DEFAULT, sf::Color::Yellow
    ));
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
        tabsys->draw(window);
        window.display();
    }

    delete tabsys;
    return 0;
}
