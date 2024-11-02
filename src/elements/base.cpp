#include <SFML/Graphics.hpp>

#include "config.hpp"
#include "base.hpp"

void SFBase::handleEvent(const sf::Event& event) {
    switch (event.type) {
        case sf::Event::MouseButtonPressed: {
            if (!isVectorInBounds(
                ROOT_WINDOW->mapPixelToCoords(
                    sf::Mouse::getPosition(*ROOT_WINDOW)
                )
            )) return;
            keyPressed(event.mouseButton.button);
            break;
        }
        case sf::Event::MouseButtonReleased: {
            if (!isVectorInBounds(
                ROOT_WINDOW->mapPixelToCoords(
                    sf::Mouse::getPosition(*ROOT_WINDOW)
                )
            )) return;
            keyReleased(event.mouseButton.button);
            break;
        }
        case sf::Event::MouseMoved: {
            mouseMoved(
                ROOT_WINDOW->mapPixelToCoords(
                    sf::Mouse::getPosition(*ROOT_WINDOW)
                )
            );
            break;
        }
    }
}

void SFBase::draw(sf::RenderTarget &target, sf::RenderStates states) const {
    throw std::logic_error("U can't call this method");
}

void SFBase::mouseMoved(const sf::Vector2f&) {}
void SFBase::keyPressed(const sf::Mouse::Button&) {}
void SFBase::keyReleased(const sf::Mouse::Button&) {}
