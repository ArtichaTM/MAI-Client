#include "tab.hpp"

#include "config.hpp"

Tab::Tab(const std::string& title, float offset, float height, sf::Color color)
    : tabText(offset, 0, title, PATH_FONT_DEFAULT, color) {}

Tab::~Tab() {
    for (SFBase*& element : elements) {
        delete element;
    }
}

void Tab::draw(sf::RenderWindow &window) const { draw(window, false); }

void Tab::draw(sf::RenderWindow& window, bool active) const {
    tabText.draw(window);
    if (!active) return;

    for (const SFBase* element : elements) {
        element->draw(window);
    }
}

void Tab::handleEvent(const sf::Event& event) {
    tabText.handleEvent(event);
    for (SFBase* element : elements) {
        element->handleEvent(event);
    }
}

bool Tab::IsSwitcherInBounds(const sf::Vector2i& position) const {
    return tabText.shape.getGlobalBounds().contains(
        ROOT_WINDOW->mapPixelToCoords(position)
    );
}

bool Tab::IsSwitcherInBounds(const sf::Vector2f& position) const {
    return tabText.shape.getGlobalBounds().contains(position);
}

void Tab::AddElement(SFBase* element) { elements.emplace_back(element); }

const std::string Tab::getName() { return tabText.text.getString(); }
